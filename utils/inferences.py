import math, time, openai
from tqdm import tqdm
from openai.error import RateLimitError, APIConnectionError, Timeout
from utils.resolver import resolver
from utils.tools import get_reduced_context_string, save_df, pred_string_to_pred_list
import datetime

# Helper function to make inference on single batch:
def infer(batch, model, verbose):

    # PROMPTS ARE AMENDED MANUALLY DEPENDING ON THE EXPERIMENT:
    # ---------------------------------------
    prompt = "You are a medical professional that expands abbreviations. Expand the medical abbreviations marked by the characters '~~' in the following sub-heading and sentence snippets from a doctor's note. Provide the expanded abbreviations in a numbered list from 1 to " + str(len(batch)) + " with numbers matching the question number marked with (NO_). In your response state only the expanded abbreviation. Do nothing else! Do not state the original abbreviation. Do not state your task:"
    for i in range(0, len(batch)):
        prompt += " (NO_" + str(i+1) + "): Subheading: '" + batch.iloc[i]['section'] + "'. Snippet: '..." + get_reduced_context_string(batch.iloc[i]['left_string'],batch.iloc[i]['right_string'],batch.iloc[i]['sf_rep'],width=100,add_marker=True) +"...'. "
    # ---------------------------------------

    if model == 'gpt-3.5-turbo':
        messages=[
            {"role": "system", "content": "You are a medical professional who expands abbreviations."},
            {"role": "user", "content": prompt}
        ]
        predictions_raw = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)['choices'][0]['message']['content'].replace("\n", " ").strip().lower()
    else:
        predictions_raw = openai.Completion.create(model=model, prompt=prompt, temperature=0, max_tokens=1028)['choices'][0]['text'].replace("\n", " ").strip()

    predictions = [predictions_raw] if len(batch) == 1 else pred_string_to_pred_list(predictions_raw)

    if verbose:
        print("PROMPT:", prompt)
        print("PREDICTIONS RAW:", predictions_raw)
        print("PREDICTIONS    :", predictions)
        print("BATCH SFS:",[batch.iloc[i]['sf'] for i in range(0,len(batch))])

    try:
        assert(len(batch) == len(predictions))
    except AssertionError:
        if verbose:
            print("**Assertion Error!***")
        raise AssertionError

    return predictions

# Helper function to save data to pickle
def save_chat_data(iter, data):
    now = datetime.datetime.now()
    now_time = now.strftime("%d_%m-%H_%M")
    save_name = './results/' + 'data_chat_' + iter + "_" + now_time + '.pkl'
    save_df(data, save_name)

# Main run function for chat inferences:
def run_inferences(data, mapper, api_key, error_indices, model, batch_size, limit, verbose=True, save_freq=1000):

    print("Running inferences on model: %s" % model)

    openai.api_key = api_key
    num_batches = math.ceil(min(len(data), limit)/batch_size)
    print("Running %d batches of size %d (total %d instances):" % (num_batches, batch_size, min(limit,len(data))))

    errors = 0

    for i in tqdm(range(0, min(len(data), limit), batch_size)):
        success = False

        while True:
            try:
                predictions = infer(data[i:i+batch_size], model, verbose)
                success = True
                break
            except RateLimitError:
                if verbose:
                    print("********* API Rate limit reached. Sleep 20 seconds and continue.**********")
                time.sleep(20)
                continue
            except APIConnectionError:
                if verbose:
                    print("********* API connection error. Sleep 5 seconds and continue.   **********")
                time.sleep(5)
                continue
            except Timeout:
                print("********* Request Time Out Error. Sleep 120 seconds and continue.   **********")
                time.sleep(120)
                continue
            except AssertionError:
                for x, _ in data[i:i+batch_size].iterrows():
                    error_indices.append(x)
                print("*********Skipped batch from %d to %d. due to assertion error:*********" % (i,i+batch_size))
                break

        if success:
            for j, prediction in enumerate(predictions):
                data.iloc[i + j, data.columns.get_loc('prediction')] = prediction
                data.iloc[i + j, data.columns.get_loc('pred_lf')],  data.iloc[i + j, data.columns.get_loc('pred_lf_idx')] = resolver(prediction, mapper[data.iloc[i + j]['sf']])
        else:
            errors += batch_size

        if i > 0 and i % save_freq == 0:
            save_chat_data(str(i), data)

    save_chat_data("end", data)

    print("Completed %d instances with %d errors." % (len(data), errors))

    return data


