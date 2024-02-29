import os
import subprocess
import os.path
import sys

hostname = "ccnt-ubuntu"

if hostname == "ccnt-ubuntu":
    print(hostname)

    def preprocess():
        log = "/Users/ahura/Nexus/code_summarization_public/log.preprocess"
        if os.path.exists(log):
            os.system("rm -rf %s" % log)

        run = (
            "python preprocess.py "
            "-data_name github-python "
            "-train_src /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code "
            "-train_tgt /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.comment "
            "-train_xe_src /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code "
            "-train_xe_tgt /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.comment "
            "-train_pg_src /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code "
            "-train_pg_tgt /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.comment "
            "-valid_src /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/dev0.60.20.2.code "
            "-valid_tgt /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/dev0.60.20.2.comment "
            "-test_src /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/test0.60.20.2.code "
            "-test_tgt /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/test0.60.20.2.comment "
            "-save_data /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/processed_all "
            "> /store/travail/vamaj/code_summarization_public/log.preprocess"
        )
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    def train_a2c(
        start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus
    ):
        run = (
            "python a2c-train.py "
            "-data /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/processed_all.train.pt "
            "-save_dir /store/travail/vamaj/Leto/src/code_sum_drl/dataset/result/ "
            "-embedding_w2v /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/ "
            "-start_reinforce %s "
            "-end_epoch %s "
            "-critic_pretrain_epochs %s "
            "-data_type %s "
            "-has_attn %s "
            "-gpus %s"
            % (
                start_reinforce,
                end_epoch,
                critic_pretrain_epochs,
                data_type,
                has_attn,
                gpus,
            )
        )
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    def test_a2c(data_type, has_attn, gpus):
        run = (
            "python a2c-train.py "
            "-data /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/processed_all.train.pt "
            "-load_from /store/travail/vamaj/Leto/src/code_sum_drl/dataset/result/model_rf_hybrid_1_30_reinforce.pt "
            "-embedding_w2v /store/travail/vamaj/Leto/src/code_sum_drl/dataset/train/ "
            "-eval -save_dir . "
            "-data_type %s "
            "-has_attn %s "
            "-gpus %s " % (data_type, has_attn, gpus)
        )
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    if sys.argv[1] == "preprocess":
        preprocess()

    if sys.argv[1] == "train_a2c":
        train_a2c(
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
        )

    if sys.argv[1] == "test_a2c":
        test_a2c(sys.argv[2], sys.argv[3], sys.argv[4])
