from kedro.pipeline import Pipeline, node

from .nodes import clean_data, split_data, validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_raw_data,
                inputs=["raw_wdbc", "params:data_validation"],
                outputs="validated_wdbc",
                name="validate_raw_data_node",
            ),
            node(
                func=clean_data,
                inputs="validated_wdbc",
                outputs="cleaned_wdbc",
                name="clean_data_node",
            ),
            node(
                func=split_data,
                inputs=["cleaned_wdbc", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )
