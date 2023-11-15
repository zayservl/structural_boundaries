# structural_boundaries
Проект для восстановления структурных границ в межпрофильном пространстве. 
Для запуска требуется запустить main.py с входным аргументом в виде json файла с параметрами работы. 
Примеры запуска лежат в папке structural_boundaries\model_presets
Для оценки качества прогноза рекомендуется запустить скрипт с параметрами model_presets\run_cross_val_areas.json
Для оценки важности признаков рекомендуется запустить model_presets\run_feature_importance_estimation.json
После этого наилучший результат для конкретного набора данных будет получен model_presets\fit_predict_FCNN.json или model_presets\fit_predict_UNET05.json
