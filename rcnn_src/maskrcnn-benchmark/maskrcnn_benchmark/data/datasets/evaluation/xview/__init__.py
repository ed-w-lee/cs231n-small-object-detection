from .xview_eval import do_xview_evaluation


def xview_evaluation(
    dataset,
    predictions,
    output_folder,
    **_
):
    return do_xview_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
    )
