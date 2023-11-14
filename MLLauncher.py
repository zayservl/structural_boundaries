import json

from MLModel import *
from MLData import *

"""
launcher_keys - ключи в JSON файле, в которых передаются параметры запуска.
task - пока единственный ключ в JSON файле, в котором передается задание для запуска.
Есть три варианта задания:
fit_predict - обучить модель и сделать прогноз на всей территории
crossval - сделать кросс-валидацию
features_importance - оценить важность признаков с помощью значений Шепли.

В любом из перечисоенных сценариев обязательно нужно передавать пути к обучающей выборке, параметры 
данных (интерполяция, макс. коэффициент корреляции), а также  параметры обучения модели. 
При fit_predict сценраии также необходимо передать путь к файлу для сохранения результатов прогноза.
"""
launcher_keys = ['task']

"""
data_keys - ключи в JSON файле, в которых передаются параметры данных.
data_constructor_keys - ключи, которые передаются в конструкторе класса MLData

Learning_Grd_Files - директория, в которой содержатся grd файлы с признаками
Target_Grd_File - путь к grd файлу с известными значениями структурного каркаса
Mesh - параметр, с помощью которого можно менять параметры сетки гридов. Передвается в виде словаря {'dx': dx, 'dy': dy}
Max_corr - числовой параметр, с помощью которого можно задать пороговое значение коэффициента корреляции, по которому
           будет происходить отбраковка неинформативных признаков.
interpolate_grids - если True, то автоматически происходит интерполяция всех гридов к единой сетке. По умолчанию true
interpoltaion_method - метод интерполяции (nearest, linear, cubic). См. scipy.interpolate.griddata.
regrid_mesh_type - параметр, который регулирует параметры сетки, к которой все гриды приводятся при интерполяции. Есть
                   3 варианта: 'to_target' - все приводится к сетке таргета, 'to_features' - все приводится к сетке признаков,
                               если признаки имеют разные параметры разбиения, то по ним автоматически ищется минимум, максимум
                               и минимальный шаг; 'min_max' - все как с признаками, только при определении пределов по координатам
                               и шага участвует еще и таргет.
"""

data_keys = ['Learning_Grd_Files', 'Target_Grd_File',
             'Output_File', 'Mesh', 'Max_corr']
data_constructor_keys = ['interpolate_grids',
                         'interpolation_method', 'regrid_mesh_type']

"""
model_keys - ключи в JSON файле, в которых передаются параметры модели машинного обучения.
Каждый параметр имеет значение по умолчанию, поэтому в JSON файле необходимо указывать только те
параметры, которые хочется настроить вручную. 

Method - текстовый параметр, задающий модель машинного обучения. Возможные варианты смотреть
         в методе set_model класса MLModel (апараметр model_type).
n_estimators - параметр, регулирующий количество деревьев в бустинге или случайном лесе.
max_depth - параметр, регулирующий максимальную глубину деревьев в бустинге или случайном лесе.
alpha - параметр регуляризации для Ridge, Lasso и ElasticNet.
l1_ratio - параметр, регулирующий соотношение коэффициентов L1 и L2 регуляризации в ElasticNet.
kernel - тип ядря для kernel-trick в SVM. возможные варианты - 'rbf', 'linear', 'polynomial'
nuerons_list - список нейроной для полносвязной нейронной сети (например, [10, 15, 20, 5])
activation - функция активации в скрытых слоях нейронных сетей. Можно указывать любое текстовое 
             наименование, доступное в TensorFlow. 
optimizer - метод оптимизации. Можно указывать любое текстовое наименование, доступное в TensorFlow.
loss - функция потерь, которая минимизируется при обучении нейронных сетей. Можно указывать любое текстовое 
       наименование, доступное в TensorFlow. 
metrics - список текстовых наименований метрик, которые пишутся в консоль после каждой эпохи обучения
          нейронной сети, а также оцениваются при кросс-валидации.
          Доступны варианты 'rmse', 'r2' и 'mpe'. Передается в виде списка, даже если метрика одна
conv_sample_shape - параметр, регулирующий размер области, по которой строится каждый отдельный прогноз
                    сверточной нейронной сети. Передается в виде списка количества ячеек по каждой оси [Nx, Ny]
n_layers - количество скрытых слоев в нейронной сети
n_filters - количество фильтров на каждый признак для первого скрытого слоя сверточных нейронных сетей
max_pooling - использовать ли макс пулинг в UNet 1/2
kernel_size - размер оператора свертки в количестве ячеек
n_neighbors - количество соседей для KNN
cache_size - размер кэша для SVM
random_state - random_state
"""

model_keys = ['Method', 'n_estimators', 'max_depth', 'alpha', 'l1_ratio', 'kernel',
              'neurons_list', 'activation', 'optimizer',
              'loss', 'metrics', 'conv_sample_shape',
              'n_layers', 'n_filters', 'max_pooling',
              'kernel_size', 'n_neighbors', 'cache_size', 'random_state']

"""
cross_val_keys - ключи в JSON файле, в которых передаются параметры кросс-валидации, если задание - crossval.

split_type - тип разбиения при кросс-валидации. Есть три варианта:
             'random' - каждая точка случайным образом попадает в одну из n_splits частей данных для кросс-валидации.
                        этот вариант недоступен для сверточных нейронных сетей.
             'areas' - территория разбивается на регионы заданного размера (параметр cell_size), после чего
                       каждый регион случайным образом попадает в одну из n_splits частей данных для кросс-валидации.
             'geographic' - территория разбивается по оси X и Y на n_splits частей (по каждой оси!), после чего 
                            последовательно каждый кусок используется в качестве отложенной выборки, а все остальные
                            для обучения
n_splits - количесво частей, на которые разбивается обучающая выборка
metrics - список текстовых наименований метрик, которые пишутся в консоль после каждой эпохи обучения
          нейронной сети, а также оцениваются при кросс-валидации.
          Доступны варианты 'rmse', 'r2' и 'mpe'. Передается в виде списка, даже если метрика одна
          
cell_size - размер региона в ячейках для 'areas' кросс-валидации. Передается в виде списка [Nx, Ny]
n_epochs - количество эпох для обучения нейронной сети. Если используете не нейронную сеть, то указывать не нужно.
           Рекомендуемое значение - от 50, хотя на некоторых данных может хватить и 5-10.
print_stats - если True, то на каждом шаге кросс-валидации в консоль печатается статистика (мин, макс, сред) по
              обучающей и валидационной выборке.
plot_maps - параметр, полезный при запуске в юпитере. Если True, то в output'е ячейки рисуются карты с распределением
            обучающей и валидационной выборке на каждом шаге кросс-валидации.
verbose - параметр, регулирующий печать в консоль при обучении нейронных сетей. см. параметр verbose в tf.keras.Model
batch_size - размер батча для обучения нейронных сетей

"""

cross_val_keys = ['split_type', 'n_splits', 'cell_size', 'n_epochs',
                  'print_stats', 'plot_maps', 'verbose', 'batch_size']

"""
fit_keys - ключи в JSON файле, в которых передаются параметры обучения модели, если задание - fit_predict.
n_epochs - количество эпох для обучения нейронной сети. Если используете не нейронную сеть, то указывать не нужно.
           Рекомендуемое значение - от 50, хотя на некоторых данных может хватить и 5-10.
verbose - параметр, регулирующий печать в консоль при обучении нейронных сетей. см. параметр verbose в tf.keras.Model
batch_size - размер батча для обучения нейронных сетей
"""

fit_keys = ['n_epochs', 'batch_size', 'verbose']

"""
feature_importance_keys - ключи в JSON файле, в которых передаются параметры оценки важности признаков, 
                          если задание - 'features_importance'.

N_samples - количество точек, по которым строится оценка. Т.к. значения Шепли вычисляются по всем
            возможным комбинациям признаков, рекомендуется не пропускать данный параметр. 
            Рекомендуется не указывать значения в десятки и сотни тысяч. На 1000 объектов 
            catboost из 1500 деревьев делает оценку около 10 минут при 20 признаках.
summary_plot - параметр, полезный при запуске в юпитере. Если True, то в output'е ячейки рисуются диаграммы
               по значениям Шепли
"""

feature_importance_keys = ['N_samples', 'summary_plot']

"""
Параметры, не упомянутые выше:
conv_samples_step - дополнительный параметр, задающий шаг окна при формировании данных для сверточных сетей.
                    Передается в виде словаря {'dx' : Nx, 'dy': Ny}
"""


def filter_keys(d: dict, k_list: list) -> dict:
    """
    Вспомогательная функция для извлечения из словаря элементов с ключами из списка
    """

    return {k: d[k] for k in k_list if k in d}


class MLLauncher():
    """
    Класс для запуска машинного обучения через командную строку.
    При создании объекта класса необоходимо указать путь к Json файлу, в котором указаны параметры.
    Возможны 3 варианта запуска:
    - Обучение и предсказание
    - Кросс-валидация
    - Оценка важности признаков
    Во всех случаях необходимо указать путь к директории с гридами, в которых содержатся признаки, и
    путь к файлу с таргетом. 
    """

    def __init__(self, json_name: str):
        """
        Конструктор, единственным аргументом является путь к Json файлу с параметрами запуска.
        """
        with open(json_name, 'r', encoding='utf-8') as fp:
            self.parameters = json.load(fp)

    def get_model_parameters(self, data: MLData) -> dict:
        """
        Вспомогательный метод, который считывает и обрабатывает параметры модели, которые 
        были указаны в файле.
        """

        model_params = filter_keys(self.parameters, model_keys)

        if 'Method' in model_params:
            model_params['model_type'] = model_params.pop('Method')
        if 'conv_sample_shape' in model_params:
            model_params['conv_sample_shape'].append(len(data.grids))
        if model_params['model_type'] == 'FCNN':
            model_params['input_shape'] = (data.X.shape[1],)
        if 'n_filters' in model_params:
            model_params['n_filters'] *= len(data.grids)

        return model_params

    def fit_predict(self, model: MLModel, data: MLData,
                    model_params: dict, fit_params: dict, data_params: dict):
        """
        Метод для обучения модели и прогноза на всю территорию.
        По результатам своей работы сохраняет грид-файл с именем, указанным
        в параметре "Output_File".
        """

        output_name = data_params['Output_File']

        if not model.is_conv():
            model.fit(data.X, data.y, **fit_params)
            y_pred = model.predict(data.X_all)
        else:
            sNx = model_params['conv_sample_shape'][0]
            sNy = model_params['conv_sample_shape'][1]

            if 'conv_samples_step' in self.parameters:
                sdx = self.parameters['conv_samples_step']['dx']
                sdy = self.parameters['conv_samples_step']['dy']
            else:
                sdx = 1
                sdy = 1

            if model.model_type == 'UNet':
                X, y, _, _ = data.get_samples_for_encoder_decoder_CNN(sNx, sNy, sdx, sdy)
            else:
                X, y, _, _ = data.get_samples_for_encoder_CNN(sNx, sNy, sdx, sdy)

            model.fit(X, y, **fit_params)

            if model.model_type == 'UNet':
                X, samples_ids, _ = data.get_samples_for_encoder_decoder_CNN(sNx, sNy, sdx, sdy, use_target=False)
                y_pred = model.predict(X)
                _, _, y_pred = data.create_grid_by_encoder_decoder_CNN_predict(y_pred, samples_ids, sNx, sNy)
            else:
                X, samples_ids, _ = data.get_samples_for_encoder_CNN(sNx, sNy, sdx, sdy, use_target=False)
                y_pred = model.predict(X)
                _, _, y_pred = data.create_grid_by_encoder_CNN_predict(y_pred, samples_ids)

        y_pred = data.untransform_predict(y_pred)
        y_pred = y_pred.reshape(data.target_shape)
        print('save ' + output_name)
        save_surfer_ascii_grid(
            output_name, data.x_coords, data.y_coords, y_pred)

    def cross_validation(self, model: MLModel, data: MLData):
        """
        Метод для запуска кросс-валидации.
        По результатам своей раблоты печатает в консоль метрики, вычисленные на каждом шаге
        кросс-валидации.
        """

        cross_val_params = filter_keys(self.parameters, cross_val_keys)
        if 'conv_samples_step' in self.parameters:
            cross_val_params['dx'] = self.parameters['conv_samples_step']['dx']
            cross_val_params['dy'] = self.parameters['conv_samples_step']['dy']
        scores, _ = model.cross_val_score(data, **cross_val_params)

    def feature_importance(self, model: MLModel, data: MLData, fit_params: dict, model_params: dict):
        """
        Метод для запуска оценки важности признаков. 
        По результатам своей работы печатает в консоль сортированный список признаков от самого важного к 
        наименее важным, для каждого признака указывается среднее влияние на глубину прогноза.
        """

        feature_importance_params = filter_keys(
            self.parameters, feature_importance_keys)

        if not model.is_conv():
            X = data.X
            y = data.y
        else:
            sNx = model_params['conv_sample_shape'][0]
            sNy = model_params['conv_sample_shape'][1]
            sdx = 1
            sdy = 1
            if 'conv_samples_step' in self.parameters:
                sdx = self.parameters['conv_samples_step']['dx']
                sdy = self.parameters['conv_samples_step']['dy']

            if model.model_type == 'UNet':
                X, y, _, _ = data.get_samples_for_encoder_decoder_CNN(sNx, sNy, sdx, sdy)
            else:
                X, y, _, _ = data.get_samples_for_encoder_CNN(sNx, sNy, sdx, sdy)

        model.fit(X, y, **fit_params)
        sorted_features, shap_values = model.calc_feature_importance(
            data, **feature_importance_params)

        df_fi = pd.DataFrame(
            columns=['Featrue Importance'], index=sorted_features)
        df_fi['Featrue Importance'] = shap_values
        print(f'Featrue Importance list: {df_fi}')

    def run(self):
        """
        Метод для запуска вычислений. Вызывать после создания объекта класса.
        """

        data_params = filter_keys(self.parameters, data_keys)
        data_constructor_params = filter_keys(
            self.parameters, data_constructor_keys)

        features_folder = data_params['Learning_Grd_Files']
        target_fname = data_params['Target_Grd_File']

        if 'Mesh' in data_params:
            data_constructor_params['cell_size'] = [
                data_params['Mesh']['dx'], data_params['Mesh']['dy']
            ]

        data = MLData(**data_constructor_params)

        data.import_target(target_fname)
        data.import_features(features_folder)
        data.preprocess_data()

        if 'Max_corr' in data_params:
            data.remove_features_by_correlation(data_params['Max_corr'])

        model_params = self.get_model_parameters(data)
        fit_params = filter_keys(self.parameters, fit_keys)

        model = MLModel()
        model.set_model(**model_params)

        launcher_params = filter_keys(self.parameters, launcher_keys)
        task = launcher_params['task'] if 'task' in launcher_params else 'fit_predict'

        if task == 'fit_predict':

            self.fit_predict(model, data, model_params,
                             fit_params, data_params)

        elif task == 'crossval':
            self.cross_validation(model, data)

        elif task == 'features_importance':
            self.feature_importance(model, data, fit_params, model_params)
