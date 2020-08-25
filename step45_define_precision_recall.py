import argparse

def is_binary_problem():
    # Default optimal precision/recall threshold
    optimal_threshold = 0.0



def set_precision_recall_threshold():
    # Get data




    # Check if precision recall can be applied, i.e. it is a binary problem
    if len(y_classes) > 2:
        print("The problem is a multi class problem. No precision/recall optimization will be done.")
    else:
        print("The problem is a binary class problem. Perform precision/recall analysis.")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.4 - Execute narrow incremental search for SVM')
    parser.add_argument("-exe", '--execute_narrow', default=True,
                        help='Execute narrow training', required=False)
    parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    #execute_wide_run(execute_search=args.execute_wide, data_input_path=args.data_path)

    # Execute narrow search
    #execute_narrow_search(data_input_path=args.data_path)

    print("=== Program end ===")