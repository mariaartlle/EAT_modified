import os, glob, time
import pandas as pd 
import soothsayer_utils as so
import urllib.request as rec


def hmmscan_deepTF(db_fasta):

    def extract_df_hmmer(file, hmmerprogram=None, hmmerformat='domtblout'):
        '''
        File converter into a dataframe. By default processes tabulated files, if indicated also processes
        tabulated HMMER output.
        '''
        if hmmerprogram != None:
            if hmmerformat != None:
                df = so.read_hmmer(file, program=hmmerprogram, format=hmmerformat)
                df.columns = ['_'.join(col) for col in df.columns.values]
                return df

    def non_overlapping_parser(hmmer_file, hmmerprogram='hmmscan'):
        '''
        When a hmmer output 'domtblout' file is provided, returns a parsed file with only the most significant non-overlapping domains.
        2 core functions:
            - check_overlapping() returns true when there is overlapping domains in a dataframe
            - remove_overlapping() iterates through a dataframe and removes the overlapping domains with lower i-Evalue
        '''
        def check_overlapping(subset):
            # reset df indexing and create and empty dictionary with coordinate numbers
            subset.reset_index(drop=True, inplace=True)
            coord_dict = dict.fromkeys(range(5000+1), [])

            for i in range(len(subset.axes[0])):
                domain = subset.at[i, 'identifier_target_name']
                from_coord = subset.at[i, 'ali_coord_from']
                to_coord = subset.at[i, 'ali_coord_to']
                coords_list = list(range(int(from_coord), int(to_coord) + 1))

                for coord in coords_list:
                    if coord_dict[coord] == []:
                        coord_dict[coord] = [domain]
                    else:
                        coord_dict[coord].append(domain)

            # iterate in dictionary and return True if overlapping exists
            for key, values in coord_dict.items():
                if len(values) > 1:
                    return True

        def remove_overlapping(subset):

            subset_new = pd.DataFrame()
            for i in range(len(subset.axes[0])):
                try:
                    subset_not_overlap = pd.DataFrame()
                    subset_overlap = pd.DataFrame()

                    from_coord = subset.at[i, 'ali_coord_from']
                    to_coord = subset.at[i, 'ali_coord_to']
                    i_value = subset.at[i, 'this_domain_i-value']  # we'll differentiate same domains for their i-Evalue
                    x = set(range(int(from_coord), int(to_coord) + 1))
                    row = subset.loc[(subset['this_domain_i-value'] == i_value) & (subset['ali_coord_from'] == from_coord) & (subset['ali_coord_to'] == to_coord)]
                    for j in range(len(subset.axes[0])):
                        if i != j:
                            from_coord1 = subset.at[j, 'ali_coord_from']
                            to_coord1 = subset.at[j, 'ali_coord_to']
                            i_value1 = subset.at[j, 'this_domain_i-value']
                            y = range(int(from_coord1), int(to_coord1))
                            inter = x.intersection(y)
                            row1 = subset.loc[(subset['this_domain_i-value'] == i_value1) & (subset['ali_coord_from'] == from_coord1) & (subset['ali_coord_to'] == to_coord1)]
                            if inter != set():
                                subset_overlap = pd.concat([subset_overlap, row1])
                                subset_overlap = pd.concat([subset_overlap, row]).drop_duplicates()

                            else:
                                subset_not_overlap = pd.concat([subset_not_overlap, row]).drop_duplicates()

                    if subset_overlap.empty == False:
                        e_value_list = [float(x) for x in subset_overlap['this_domain_i-value']]
                        e_value = min(e_value_list)
                        if e_value_list.count(e_value) > 1:
                            subset1 = subset_overlap.loc[subset_overlap['this_domain_i-value'] == str(e_value)]
                            score_list = [float(x) for x in subset1['this_domain_score']]
                            max_score = max(score_list)
                            min_row = subset1.loc[(subset1['this_domain_i-value'] == str(e_value)) & (subset1['this_domain_score'] == str(max_score))]
                        else:
                            min_row = subset_overlap.loc[subset_overlap['this_domain_i-value'] == str(e_value)]
                        subset_new = pd.concat([subset_new, min_row]).drop_duplicates()

                    flt = ['this_domain_i-value', 'ali_coord_from', 'ali_coord_to']
                    if ((subset_not_overlap[flt] == row[flt]).all(axis=1)).sum() != 0 and subset_overlap.empty == True:
                        subset.drop(i, inplace=True)
                        subset_new = pd.concat([subset_new, subset_not_overlap])

                    subset.reset_index(drop=True, inplace=True)
                except:
                    break
            return subset_new

        df = extract_df_hmmer(hmmer_file, hmmerprogram=hmmerprogram, hmmerformat='domtblout')
        hmmer_parsed = pd.DataFrame()

        # make a unique ordered list with all the queries for every TF
        queries = set()
        queries_list = [x for x in df['identifier_query_name']]
        queries_list = [x for x in queries_list if x not in queries and queries.add(x) is None]

        for query in queries_list:
            subset = df.loc[df['identifier_query_name'] == query]  # make subsets for every TF (process whole hmmer file
            subset.reset_index(drop=True, inplace=True)

            while check_overlapping(subset) == True:
                subset = remove_overlapping(subset)

            hmmer_parsed = pd.concat([hmmer_parsed, subset])

        hmmer_parsed.reset_index(drop=True, inplace=True)
        hmmer_parsed.to_csv('{}_output_parsed.tsv'.format(hmmerprogram), sep='\t', index=False)

        return '{}_output_parsed.tsv'.format(hmmerprogram)

    if os.path.isdir('1_HMMER') == False:
        os.mkdir('1_HMMER')    
    os.chdir('1_HMMER')
    current_directory = os.getcwd()+'/'

    # 1. Prepare the Pfam database
    if os.path.exists(current_directory+'Pfam-A.hmm') == False:
        # Download PFAM database
        print('Downloading Pfam-A database...')
        rec.urlretrieve('https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz', '{}Pfam-A.hmm.gz'.format(current_directory))
        os.system('gunzip {}Pfam-A.hmm.gz'.format(current_directory))
    else:
            print('Pfam-A database found, using this local version to execute hmmscan.')

    # 1.1 Press the PFAM database
    for f in [current_directory+'Pfam-A.hmm.h3'+str(x) for x in ['i', 'p', 'm', 'f']]:
        if os.path.exists(f) == False:
            if glob.glob('*.h3*') != []:
                os.system('rm *.h3*')
            os.system('hmmpress {}Pfam-A.hmm'.format(current_directory))

        # 2. hmmscan of db_fasta
    if os.path.exists(current_directory+'/'+'hmmscan_output_parsed.tsv') == True:
        print('hmmscan file found (version: {}), proceding with hmmsearch'.format(time.strftime('%m/%d/%Y|%H:%M:%S', time.gmtime(os.path.getmtime(current_directory+'/'+'hmmscan_output_parsed.tsv')))))
        hmmscan_file = current_directory+'/'+'hmmscan_output_parsed.tsv'
    else: 
        print('Starting hmmscan...')
        os.system('hmmscan -E 0.001 --domtblout {}hmmscan_out.tsv {}Pfam-A.hmm {} > hmmscan.out'.format(current_directory, current_directory, db_fasta))
        print('hmmscan of {} finished.'.format(db_fasta.split('/')[-1]))

        # 2.1 Parse the output hmmscan file and remove overlapping domains with lower e-values
        hmmscan_file = non_overlapping_parser('hmmscan_out.tsv')
        print('Parsing of hmmscan finished.')


def parsing_output(hmmscan_parsed_file): 

    df = pd.read_csv(hmmscan_parsed_file, sep='\t')
    newdf = pd.DataFrame()

    newdf['UniprotID'] = df['identifier_query_name'].apply(lambda x: x.split('|')[1])
    newdf['Domains'] = df['identifier_target_name']
    newdf['Domain_description'] = df['identifier_query_description']
    
    
    newdf.to_csv('hmmscan_families.tsv', sep='\t', index=False)


def compare_EAT_HBI(hmmscan_file, EAT_results):
    EAT = pd.read_csv(EAT_results, sep='\t')
    hmmscan = pd.read_csv(hmmscan_file, sep='\t')
    
    hmmscan_wo_repeats = hmmscan.drop(['Domains', 'Domain_description'], axis=1).drop_duplicates()

    EAT['UniprotID'] = EAT['UniprotID_query']

    df = EAT.merge(hmmscan_wo_repeats.reset_index(), on='UniprotID')

    print(set(EAT['UniprotID']).difference(set(hmmscan['UniprotID'])))

    df.to_csv('comparison_HBI_EAT.tsv', sep='\t', index=False)


if __name__ == '__main__':
    deepTF = '/home/maria/EAT_modified/results/hmmscan/deepTF_acnes.fasta'
    hmmscan_parsed_file = '/home/maria/EAT_modified/results/hmmscan/1_HMMER/hmmscan_output_parsed.tsv'

    # parsing_output(hmmscan_parsed_file)

    EAT_results = '/home/maria/EAT_modified/results/eat_results.tsv'
    hmmscan_file = '/home/maria/EAT_modified/results/hmmscan/hmmscan_families.tsv'

    compare_EAT_HBI(hmmscan_file, EAT_results)