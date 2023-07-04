# Main resolver function:
def resolver(target, string_list):
    # Initialize variables
    longest_match_length = 0
    longest_match_index = -1

    # Iterate over the strings in the list
    for i, string in enumerate(string_list):
        table = [[0] * (len(string) + 1) for _ in range(len(target) + 1)]

        for row in range(1, len(target) + 1):
            for col in range(1, len(string) + 1):
                if target[row - 1] == string[col - 1]:
                    table[row][col] = table[row - 1][col - 1] + 1
                    if table[row][col] > longest_match_length:
                        longest_match_length = table[row][col]
                        longest_match_index = i

    return string_list[longest_match_index], int(longest_match_index)