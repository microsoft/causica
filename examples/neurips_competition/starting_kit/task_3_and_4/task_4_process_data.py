from examples.neurips_competition.starting_kit.task_3_and_4.task_4_util import (
    get_parser_proc_data,
    load_and_process_eedi_data,
)


def main():
    parser = get_parser_proc_data()
    args = parser.parse_args()
    load_and_process_eedi_data(args.data_path, args.save_dir)


if __name__ == "__main__":
    main()
