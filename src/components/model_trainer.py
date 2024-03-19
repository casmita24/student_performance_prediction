import os
import sys

from dataclasses import dataclass

from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.components.model_parameters import ModelsConfig, ModelParametersConfig

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models_config = ModelsConfig()
        self.model_parameters_config = ModelParametersConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = self.models_config.models
            params = self.model_parameters_config.params

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No besrt model found')
            
            logging.info(f'Best model found: {best_model}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted_result = best_model.predict(X_test)

            r2_score_ = r2_score(y_test, predicted_result)

            return r2_score_
        
        except Exception as e:
            raise CustomException(e, sys)