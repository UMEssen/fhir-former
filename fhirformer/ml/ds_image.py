# from fhirformer.helper.util import timed
#
#
# class ICDCodeDataset:
#     pass
#
#
# class DS_Task_Image_Predict:
#     def __init__(
#         self,
#         config,
#         model_checkpoint: str,
#         batch_size: int = 2,
#         epochs: int = 2,
#         train_ratio: float = 0.8,
#         prediction_cutoff: float = 0.5,
#     ):
#         super().__init__(
#             config=config,
#             dataset_class=ImageCodeDataset,
#             dataset_args={"max_length": None, "num_samples": None},
#             model_checkpoint=model_checkpoint,
#             batch_size=batch_size,
#             epochs=epochs,
#             train_ratio=train_ratio,
#             prediction_cutoff=prediction_cutoff,
#         )
#         label_freq = np.sum(
#             self.dataset.labels, axis=0
#         )  # sum over the column (each label)
#         self.top10_classes = label_freq.argsort()[-10:][
#             ::-1
#         ]  # find top 10 most common classes
#
#         self.model.config.problem_type = (
#             "multi_label_classification"  # specify problem type
#         )
#         self.model.loss = BCEWithLogitsLoss()  # specify loss function for multi-label
#
#         labels = [label for label in self.train_dataset[0]["label_codes"]]
#         self.set_id_to_label(labels)
#
#     def train(self):
#         pass
#
#
# @timed
# def main(config):
#     print("Hello World!")
#     ds_image = DS_Task_Image_Predict(
#         config, config["model_checkpoint"], epochs=config["train_epochs"]
#     )
#     ds_image.train()
