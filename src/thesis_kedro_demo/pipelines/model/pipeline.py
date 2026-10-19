from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model_options",
                    "params:random_state",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
