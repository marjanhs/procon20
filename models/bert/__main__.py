from models.bert import __main__main, personality_analyze, transfer_learning, classify
from models.bert.args import get_args

if __name__ == "__main__":
    args = get_args()
    if args.do == 'main':
        print("Performing Regular Main function")
        __main__main.do_main()
    elif args.do == 'analyze':
        print('Performing personality Analyze')
        personality_analyze.do_main()
    elif args.do == 'transfer':
        print('Performing Transfer Learning!')
        transfer_learning.do_main()
    elif args.do == "classify":
        print('Performing classify')
        classify.do_main()

