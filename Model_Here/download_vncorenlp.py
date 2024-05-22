# import py_vncorenlp
# import os

# vncorenlp_path = "./VN_core_NLP"

# if not os.path.exists(vncorenlp_path):
#     os.makedirs(vncorenlp_path)

# # Automatically download VnCoreNLP components from the original repository
# # and save them in some local machine folder
# py_vncorenlp.download_model(save_dir='./VN_core_NLP')

# # Load the word and sentence segmentation component
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/content/VN_core_NLP')
# model = py_vncorenlp.VnCoreNLP(save_dir='./VN_core_NLP')