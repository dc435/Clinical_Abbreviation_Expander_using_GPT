import nltk, string, math, pickle, re
import pandas as pd
import datetime

stop_words = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)
punctuations.remove("~")

# This is the list of shortforms used by Adams de-noising (and my us to obtain the main_set):
LMC_sfs=['AMA', 'ASA', 'AV', 'BAL', 'BM', 'C&S', 'CEA', 'CR', 'CTA', 'CVA', 'CVP', 'DC', 'DIP', 'DM', 'DT', 'EC', 'ER', 'FSH', 'IA', 'IM', 'LA',
         'LE', 'MR', 'MS', 'OP', 'PA', 'PCP', 'PDA', 'PM', 'PR', 'PT', 'RA', 'RT', 'SA', 'SBP', 'US', 'VBG', 'CVS', 'NAD', 'NP', 'VAD']

# This is the list of instances forming the hard_set:
hard_list = [2050, 18438, 24583, 24590, 10772, 21014, 9242, 15900, 17949, 24607, 5153, 10787, 7715, 21029, 7206, 552, 23081, 7722, 5166, 5169, 15926, 11319, 5174, 21560, 15931, 23099, 580, 10821, 5188, 3142, 10822, 18502, 9299, 25173, 5215, 9311, 23649, 23140, 24676, 10857, 10858, 23146, 7791, 19056, 3705, 25723, 15483, 18043, 10877, 19073, 7816, 11914, 17034, 18060, 10382, 5264, 8851, 7830, 2199, 23707, 5283, 21668, 3747, 17065, 4778, 4782, 10928, 16050, 23223, 23224, 6848, 9408, 5314, 7364, 10949, 23751, 18633, 16073, 21710, 26832, 7378, 19672, 5342, 7400, 21739, 8428, 5373, 15616, 18694, 23311, 3347, 21781, 5398, 18712, 15640, 9498, 6427, 3873, 24869, 7464, 25897, 6955, 6445, 1840, 20785, 8498, 820, 11575, 24891, 13119, 25407, 7999, 10049, 11076, 8522, 10571, 26444, 19285, 19286, 21845, 9563, 18267, 7524, 26468, 870, 5481, 3945, 24939, 20845, 5492, 5494, 11136, 21896, 7561, 21903, 27023, 9117, 10657, 23970, 9636, 22950, 9127, 8104, 15783, 941, 18354, 3507, 25524, 1975, 10171, 23999, 11202, 19394, 7621, 15301, 18375, 21960, 23499, 15309, 9168, 27088, 24016, 13779, 6609, 9683, 7638, 18388, 23000, 15832, 16857, 23003, 7132, 5085, 10717, 7640, 23523, 10724, 5092, 8679, 9192, 21991, 15338, 5100, 10734, 27119, 5113, 25597, 8195, 2080, 2086, 2090, 18477, 10291, 10306, 24646, 18507, 24656, 18527, 10348, 2160]

# Helper functions, left right:
def get_left(text, start_idx):
    left = text[0:start_idx]
    # left = nltk.word_tokenize(left.lower())
    left = nltk.word_tokenize(left)
    return left

def get_left_tokenized(left):
    left_tokenized = [t for t in left if (t not in punctuations) and (t not in stop_words)]
    return left_tokenized

def get_right(text, end_idx):
    right = text[end_idx+1:]
    # right = nltk.word_tokenize(right.lower())
    right = nltk.word_tokenize(right)
    return right

def get_right_tokenized(right):
    right_tokenized = [t for t in right if (t not in punctuations) and (t not in stop_words)]
    return right_tokenized

def get_left_string(text, start_idx):
    return text[0:start_idx]

def get_right_string(text, end_idx):
    return text[end_idx+1:]

# Returns full set from Adams, which is derived from CASI but has already had preprocessing.
# option 'long=False' used for testing rare sf's (where 'trimmed_tokens' field not available, and not used)
def get_full_set(path, long=True):

    print("Getting full_set ...", end='')
    full_set = pd.read_csv(path)
    full_set['section'] = full_set['section'].astype(str)
    if long:
        full_set['trimmed_tokens'] = full_set['trimmed_tokens'].astype(str)
    full_set['target_lf_idx'] = full_set['target_lf_idx'].astype(int)
    full_set['left'] = full_set.apply(lambda row: get_left(row['context'], int(row['start_idx'])), axis=1)
    full_set['left_tokenized'] = full_set.apply(lambda row: get_left_tokenized(row['left']), axis=1)
    full_set['right'] = full_set.apply(lambda row: get_right(row['context'], int(row['end_idx'])), axis=1)
    full_set['right_tokenized'] = full_set.apply(lambda row: get_right_tokenized(row['right']), axis=1)
    full_set['left_string'] = full_set.apply(lambda row: get_left_string(row['context'], int(row['start_idx'])), axis=1)
    full_set['right_string'] = full_set.apply(lambda row: get_right_string(row['context'], int(row['end_idx'])), axis=1)
    full_set['prediction'] = pd.Series(dtype=str)
    full_set['pred_lf'] = pd.Series(dtype=str)
    full_set['pred_lf_idx'] = pd.Series(dtype=int)
    full_set = full_set.sample(frac=1, random_state=42)
    print("done. Length of full_set:", len(full_set))

    return full_set

# Returns main set, being all items used in Adams. Filtering of full_set based on shortform list.
def get_main_set(full_set):

    print("Getting test_set ...", end='')

    main_set = full_set[full_set['sf'].isin(LMC_sfs)]
    print("done. Length of test_set:", len(main_set))

    return main_set

# Returns subset of test_set containing all LF expansions in approx equal ratio to test_set. For development purposes.
def get_dev_set(test_set, divider):

    print("Getting dev_set: ...", end='')
    class_frequencies = test_set['target_lf'].value_counts()
    dev_set = pd.DataFrame()
    for class_label, frequency in class_frequencies.items():
        number_added = math.ceil(frequency/divider)
        class_subset = test_set[test_set['target_lf'] == class_label].sample(n=number_added, random_state=42)
        dev_set = pd.concat([dev_set, class_subset])
    dev_set = dev_set.sample(frac=1, random_state=42)
    print("done.")
    print("Length of test set              :", len(test_set))
    print("Number of long forms in test_set:", len(test_set.groupby('target_lf')))
    print("Length of dev set               :", len(dev_set))
    print("Number of long forms in dev_set :", len(dev_set.groupby('target_lf')))

    return dev_set

# Helper function for get_map. Simple preprocessing on lf options, to make matching easier.
def preprocess_mapping(text):

    tokens = nltk.word_tokenize(text.lower())
    tokens_clean = [t.lower() for t in tokens if t not in punctuations]
    return ' '.join(tokens_clean)

# Return SF-LF map dictionary
def get_mapper(path):

    print("Getting map ...", end='')
    labeled_sf_sf_map = pd.read_csv(path)
    mapper = {x: [] for x in LMC_sfs}
    count_lf = 0
    for _, row in labeled_sf_sf_map.iterrows():
        if row['sf'] in mapper:
            mapper[row['sf']].append(preprocess_mapping(row['target_lf_sense'] +" " + row['target_label']))
            count_lf += 1
    print("done.")
    print("Number of shortforms       :", len(mapper))
    print("Number of mapped longforms :", str(count_lf))

    return mapper

# Save and load dataframes:
def save_df(df, path):

    print("Saving df to pickle (%s):..." % path, end="")
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print("done.")

def load_df(path):

    print("Loading df from pickle...", end="")
    with open(path, 'rb') as f:
        df = pickle.load(f)
    print("done.")
    return df

def check_wrongs(data):
    wrongs = data[data['pred_lf_idx'] != data['target_lf_idx']]
    print("Data contains %d / %d wrong predictions." % (len(wrongs), len(data)))

# Return basic accuracy metrics:
def get_accuracy(data):
    from sklearn.metrics import accuracy_score
    try:
        acc = accuracy_score(data['target_lf_idx'],data['pred_lf_idx'])
    except ValueError:
        acc = 0
        print("ERROR: Some NaN values persist. Cannot calculate accuracy.")
    correct = data[data['pred_lf_idx'] == data['target_lf_idx']]
    wrongs = data[data['pred_lf_idx'] != data['target_lf_idx']]
    print("Data contains %d / %d correct predictions." % (len(correct), len(data)))
    print("Data contains %d / %d wrong predictions." % (len(wrongs), len(data)))
    print("Accuracy: %0.3f" % acc)

# Clears prediction column in dataframe for running a new set of inferences:
def clear_predictions(data):
    for i, _ in data.iterrows():
        data.loc[i,'prediction'] = None
        data.loc[i,'pred_lf'] = None
        data.loc[i,'pred_lf_idx'] = None
    print("Predictions cleared from data.")

def get_reduced_context(left,right,sf_rep,window,add_marker=False):
    start = left[-window:] if window <= len(left) else left
    end = right[:window] if window <= len(right) else right
    if add_marker:
        whole_list = start + ["~~"+sf_rep+"~~"] + end
    else:
        whole_list = start + [sf_rep] + end
    return ' '.join(whole_list)

# Helper function to get a reduced context around a shortform as a string:
def get_reduced_context_string(left,right,sf_rep,width,add_marker=False):
    left_width = width
    while left_width < len(left):
        if left[-left_width] != " ":
            left_width += 1
        else:
            break
    right_width = width
    while right_width < len(right):
        if right[right_width] != " ":
            right_width += 1
        else:
            break
    start = left[-left_width:] if left_width <= len(left) else left
    end = right[:right_width] if right_width <= len(right) else right
    # start = left[-width:] if width <= len(left) else left
    # end = right[:width] if width <= len(right) else right
    if add_marker:
        full_string = start + " ~~" + sf_rep + "~~ " + end
    else:
        full_string = start + " " + sf_rep + " " + end
    return full_string.strip()

# Build results csv
def build_results(data,mapper,baseline=False):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    results = pd.DataFrame(columns=['sf','count','targets','acc','mPr','mR','mF1','wPr','wR','wF1'])
    r = 4   #rounding precision
    sfs = data['sf'].unique()
    sfs.sort()

    field = 'baseline' if baseline else 'pred_lf_idx'

    try:
        for sf in sfs:
            sf_data = data[data['sf']==sf]
            count = len(sf_data)
            targets = len(mapper[sf])
            acc = round(accuracy_score(sf_data['target_lf_idx'], sf_data[field]),r)
            mPr = round(precision_score(sf_data['target_lf_idx'], sf_data[field],average='macro',zero_division=0),4)
            mR = round(recall_score(sf_data['target_lf_idx'], sf_data[field],average='macro',zero_division=0),4)
            mF1 = round(f1_score(sf_data['target_lf_idx'], sf_data[field],average='macro',zero_division=0),4)
            wPr = round(precision_score(sf_data['target_lf_idx'], sf_data[field],average='weighted',zero_division=0),4)
            wR = round(recall_score(sf_data['target_lf_idx'], sf_data[field],average='weighted',zero_division=0),4)
            wF1 = round(f1_score(sf_data['target_lf_idx'], sf_data[field],average='weighted',zero_division=0),4)
            new_row = {
                'sf':sf,
                'count':count,
                'targets':targets,
                'acc':acc,
                'mPr':mPr,
                'mR':mR,
                'mF1':mF1,
                'wPr':wPr,
                'wR':wR,
                'wF1':wF1
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        tot_count = results['count'].sum()
        tot_targets = results['targets'].sum()
        mean_acc = round(results['acc'].mean(),r)
        mean_mPr = round(results['mPr'].mean(),r)
        mean_mR = round(results['mR'].mean(),r)
        mean_mF1 = round(results['mF1'].mean(),r)
        mean_wPr = round(results['wPr'].mean(),r)
        mean_wR = round(results['wR'].mean(),r)
        mean_wF1 = round(results['wF1'].mean(),r)

        tot_row = {
            'sf':'total/mean:',
            'count':tot_count,
            'targets':tot_targets,
            'acc':mean_acc,
            'mPr':mean_mPr,
            'mR':mean_mR,
            'mF1':mean_mF1,
            'wPr':mean_wPr,
            'wR':mean_wR,
            'wF1':mean_wF1
        }

        results = pd.concat([results, pd.DataFrame([tot_row])], ignore_index=True)

        acc_overall = round(accuracy_score(data['target_lf_idx'], data[field]),r)

        now = datetime.datetime.now()
        now_time = now.strftime("%d_%m-%H_%M")
        save_name = './results/' + 'results_' + "-" + now_time + '.csv'
        results.to_csv(save_name)

        print("Saved results of %d instances and %d sfs in results csv to (%s)." % (tot_count, len(sfs), save_name))
        print("Overall accuracy: %.4f" % acc_overall)

    except ValueError:
        print("ERROR: Some NaN values persist. Cannot calculate results.")

    return results

# Helper function to convert string of raw predictions to list (prior to sending to resolver)
def pred_string_to_pred_list(predictions_raw):

    by_dots = re.findall(r'\d{1,2}\.\s', predictions_raw)
    by_brackets = re.findall(r'\(NO_\d{1,2}\)', predictions_raw)
    if len(by_brackets) > len(by_dots):
        predictions_raw = re.sub(r'\(NO_\d{1,2}\)', "###", predictions_raw)
    else:
        predictions_raw = re.sub(r'\d{1,2}\.\s', "###", predictions_raw)
    predictions = predictions_raw.split("###")
    predictions = predictions[1:]
    predictions = [p.strip() for p in predictions]
    for i,p in enumerate(predictions):
        tokens = nltk.word_tokenize(p.lower())
        predictions[i] = ' '.join([t.lower() for t in tokens if t not in punctuations])

    return predictions

# Helper function to copy corrected errors from temp dataframe to main dataframe:
def copy_corrections(data_errors,data):
    print("Copying corrected data to main dataframe.")
    for idx, _ in data_errors.iterrows():
        data.loc[idx,'prediction'] = data_errors.loc[idx]['prediction']
        data.loc[idx,'pred_lf'] = data_errors.loc[idx]['pred_lf']
        data.loc[idx,'pred_lf_idx'] = data_errors.loc[idx]['pred_lf_idx']

# Adds a 0-R baseline to the dataframe:
def add_baseline(data):

    baseline = {}
    sfs = data['sf'].unique()
    sfs.sort()

    for sf in sfs:
        sf_data = data[data['sf']==sf]
        most_common = sf_data['target_lf_idx'].value_counts().idxmax()
        baseline[sf] = most_common

    data['baseline'] = data.apply(lambda row: baseline[row['sf']], axis= 1)

# Build a CSV limited to FSH = 'Fairview Southdale Hospital' entries
def build_rare_data_csv():

    filename = "./data/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt"
    data = pd.read_csv(filename, sep='|', header=None, encoding='windows-1252', keep_default_na=False)
    data.columns = ['sf','target_lf','sf_rep','start_idx','end_idx','section','context']
    target = 'Fairview Southdale Hospital'
    rare_data = data[data['target_lf']==target]
    rare_data['target_lf'].unique()
    rare_data['target_lf_idx'] = 0
    pattern = r'(Fairview|Southdale|Hospital|hospital)'
    print(len(rare_data))
    rare_data = rare_data[~rare_data['context'].str.contains(pattern)]
    print(len(rare_data))
    rare_data.to_csv('./data/rare_data.csv')

# Return a mapper for rare_set testing
def get_rare_mapper():
    return {'FSH': ['Fairview Southdale Hospital', 'follicle-stimulating hormone', 'fascioscapulohumeral muscular dystrophy']}

