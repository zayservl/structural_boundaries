import shap
import tensorflow as tf

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from MLData import *
from keras import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dense, Flatten
from keras.layers import Conv2DTranspose

layer = tf.keras.layers.Layer
model = tf.keras.models.Model


def generate_regions_map(data_shape: list, cell_size: list) -> Tuple[ndarray, ndarray]:
    """
    Функция для создания карты регионов для кросс-валидации.
    В разультате работы функции создается 2D матрица размера data_shape,
    которая заполняется прямоугольными областями cell_size, в каждой 
    прямоугольной области в качестве значения задается уникальный номер
    данной области.
    Также вместе с картой регионов на выходе функция выдает список номеров регионов
    """

    N0 = int(np.ceil(data_shape[0] / cell_size[0]))
    N1 = int(np.ceil(data_shape[1] / cell_size[1]))

    regions = np.repeat(np.arange(N1), cell_size[1]).reshape([1, -1])
    regions = np.repeat(regions, data_shape[0], axis=0)

    for i in range(N0):
        regions[i * cell_size[0]: (i + 1) * cell_size[0], :] += i * (cell_size[0] - 1)

    return regions[:data_shape[0], :data_shape[1]], np.arange(0, regions[-1, -1] + 1)


def cellsKFold(x_all: np.ndarray, y_all: np.ndarray, n_splits: int, grid_shape: list,
               cell_size: list, random_state: int = 123):
    """
    Генератор для разделения выборки на train и test при кросс-валидации по регионам.
    Разибвает всю территорию на прямоугольные области размера cell_size и случайным
    образом распределяет часть регионов в train, а оставшуюся часть в test.
    Доли данных, попадающих в train и test можно регулировать с помощью параметра
    n_splits (количество разделений и, соответственно, шагов кросс-валидации.
    """

    regions_map, regions_list = generate_regions_map(grid_shape, cell_size)
    XY = np.hstack([x_all, y_all.reshape([-1, 1])])

    kf = KFold(n_splits, shuffle=True, random_state=random_state)

    for train_regions, test_regions in kf.split(regions_list):
        train_mask = np.zeros(grid_shape).astype('bool')
        test_mask = np.zeros(grid_shape).astype('bool')

        for region in train_regions:
            train_mask = np.logical_or(train_mask, regions_map == region)

        for region in test_regions:
            test_mask = np.logical_or(test_mask, regions_map == region)

        train_mask = np.logical_and(
            train_mask.flatten(), ~np.isnan(XY).any(axis=1))
        test_mask = np.logical_and(
            test_mask.flatten(), ~np.isnan(XY).any(axis=1))

        if train_mask.any() and test_mask.any():
            yield train_mask, test_mask


def geographicKFold(x_all: ndarray, y_all: ndarray, n_splits: int, grid_shape: tuple):
    """
    Генератор маск для обучения и теста при кросс-валидации c разделением всей территории
    на n_splits участков по осям Х и У. 
    """

    Ny = grid_shape[0]
    Nx = grid_shape[1]

    xc = np.arange(Nx)
    yc = np.arange(Ny)

    XY = np.hstack([x_all, y_all.reshape([-1, 1])])

    for i in range(n_splits):
        i1 = int(Ny * i / n_splits)
        i2 = int(Ny * (i + 1) / n_splits)
        for j in range(n_splits):
            j1 = int(Nx * j / n_splits)
            j2 = int(Nx * (j + 1) / n_splits)

            mask_test = np.zeros(grid_shape, dtype=bool)
            mask_test[i1:i2, j1:j2] = True

            mask_test = mask_test.reshape([-1])
            mask_train = ~mask_test

            mask_test = np.logical_and(mask_test, ~np.isnan(XY).any(axis=1))
            mask_train = np.logical_and(mask_train, ~np.isnan(XY).any(axis=1))

            if mask_test.any() and mask_train.any():
                yield mask_train, mask_test


def create_encoder_block(prev_layer_input: layer, n_filters: int = 32, activation: str = 'relu',
                         max_pooling: bool = True, kernel_size: int = 3) -> Tuple[layer, layer]:
    """
    Вспомогательная функция для создания слоя энкодера в нейронной сети UNet
    """

    conv = Conv2D(n_filters, kernel_size, activation=activation, padding='same',
                  kernel_initializer='HeNormal')(prev_layer_input)

    conv = Conv2D(n_filters, kernel_size, activation=activation, padding='same',
                  kernel_initializer='HeNormal')(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def create_decoder_block(prev_layer_input: layer, skip_layer_input: layer,
                         n_filters: int = 32, activation: str = 'relu', kernel_size: int = 3) -> layer:
    """
    Вспомогательная функция для создания слоя декодера в нейронной сети UNet
    """

    up = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2, 2),
                         padding='same')(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(n_filters, kernel_size, activation=activation, padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters, kernel_size, activation=activation, padding='same',
                  kernel_initializer='HeNormal')(conv)
    return conv


def create_unet(input_size: tuple = (64, 64, 3), n_layers: int = 4, n_filters: int = 32,
                activation: str = 'relu', kernel_size: int = 3, random_state: int = 123) -> model:
    """
    Функция для создания сверточной нейронной сети архитектуры U-Net.
    n_layers - количество скрытых слоев в энкодере (в декодере их будет столько же).
    в декодере в каждом слое присходит уменьшение размера входа в 2 раза по каждой оси за счет пулинга,
    поэтому количество слоев не может быть больше, чем min(log2(Nx), log2(Ny)).
    n_filters - количество фильтров в первом слое.
    activation - функция активации (см. tf.keras.activations)
    kernel_size - размер ядра свертки в ячейках
    random_state - параметр фиксации генератора случайных чисел
    """

    if n_layers > np.min(np.log2(input_size[:2])):
        raise Exception(
            "количество слоев не может быть больше, чем min(log2(Nx), log2(Ny))")

    tf.random.set_seed(random_state)
    input_data = Input(input_size)

    encoder_blocks = []
    decoder_blocks = []

    encoder_blocks.append(create_encoder_block(
        input_data, n_filters, activation, True, kernel_size))

    nf = n_filters * 2
    for i in range(n_layers - 1):
        encoder_blocks.append(create_encoder_block(
            encoder_blocks[i][0], nf, activation, True, kernel_size))
        nf *= 2

    encoder_blocks.append(create_encoder_block(
        encoder_blocks[-1][0], nf, activation, False, kernel_size))

    nf /= 2
    decoder_blocks.append(
        create_decoder_block(encoder_blocks[-1][0], encoder_blocks[-2][1], nf, activation, kernel_size))

    for i in range(len(encoder_blocks) - 2):
        nf /= 2
        decoder_blocks.append(
            create_decoder_block(decoder_blocks[-1], encoder_blocks[-i - 3][1], nf, activation, kernel_size))

    layer_last = Conv2D(n_filters, kernel_size, activation='linear', padding='same',
                        kernel_initializer='he_normal')(decoder_blocks[-1])

    output = Conv2D(1, 1, padding='same', activation='linear')(layer_last)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=Adam(), loss='mse', metrics=['RootMeanSquaredError'])
    print(model.summary())

    return model


def create_unet_05(input_size: tuple = (64, 64, 3), n_layers: int = 4, n_filters: int = 32, activation: str = 'relu',
                   max_pooling: bool = True, kernel_size: int = 3, random_state: int = 123) -> model:
    """
    Функция для создания сверточной нейронной сети архитектуры U-Net без декодера (первая половина от U-Net).
    n_layers - количество скрытых слоев без учета полносвязного слоя в конце. Если используется пулинг, то 
    количество слоев не может быть больше, чем min(log2(Nx), log2(Ny)).
    n_filters - количество фильтров в первом слое.
    activation - функция активации (см. tf.keras.activations)
    kernel_size - размер ядра свертки в ячейках
    random_state - параметр фиксации генератора случайных чисел
    """

    tf.random.set_seed(random_state)

    model = tf.keras.models.Sequential()
    nf = n_filters

    model.add(Conv2D(nf, (kernel_size, kernel_size), activation=activation,
                     padding='same', input_shape=input_size))
    model.add(Conv2D(nf, (kernel_size, kernel_size), activation=activation,
                     padding='same'))

    for i in range(n_layers - 1):
        nf *= 2
        model.add(Conv2D(nf, (kernel_size, kernel_size),
                         activation=activation, padding='same'))
        model.add(Conv2D(nf, (kernel_size, kernel_size),
                         activation=activation, padding='same'))

        if max_pooling:
            model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    nf *= 2
    model.add(Dense(nf, activation=activation))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(),
                  loss='mse', metrics=['RootMeanSquaredError'])
    print(model.summary())

    return model


def create_fully_connected_nn(input_shape: tuple, neurons_list: list, activation: str = 'relu',
                              loss: str = 'mse', random_state: int = 123) -> model:
    """
    Функция для создания полносвязной нейронной сети.
    neurons_list - список нейронов для каждого скрытого слоя. Количество скрытых слоев 
    определяется исходя из количества значений в данном списке
    activation - функция активации (см. tf.keras.activations)
    """

    tf.random.set_seed(random_state)

    model = tf.keras.models.Sequential([tf.keras.Input(shape=input_shape)])

    for N_neurons in neurons_list:
        model.add(Dense(N_neurons, activation=activation,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)))

    model.add(Dense(1, activation='linear',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)))

    model.compile(optimizer='adam', loss=loss, metrics=['RootMeanSquaredError'])
    print(model.summary())

    return model


class MLModel:
    """
    Класс модели машинного обучения.
    В конструкторе необходимо передать параметры модели. Все аргументы именованные и передавать нужно только
    те аргументы, которые используются для данного типа модели.
    """

    def __init__(self):
        """
        Конструктор. Не принимает никаких аргументов и создает модель с параметрами по умолчанию.
        """
        self.n_estimators = None
        self.model_type = None
        self.model = None
        self.max_depth = None
        self.alpha = None
        self.l1_ratio = None
        self.kernel = None
        self.neurons_list = None
        self.activation = None
        self.loss = None
        self.metrics = None
        self.input_shape = None
        self.conv_sample_shape = None
        self.n_layers = None
        self.n_filters = None
        self.kernel_size = None
        self.n_neighbors = None
        self.cache_size = None
        self.random_state = None
        # self.set_model()

    def set_model(self, model_type: str = 'RF', n_estimators: int = 100, max_depth: int = None,
                  alpha: float = 1, l1_ratio: float = 0.5, kernel: str = 'rbf',
                  neurons_list=None, activation: str = 'relu',
                  loss: str = 'mse', metrics=None, input_shape: tuple = None,
                  conv_sample_shape=None, n_layers: int = 2, n_filters: int = None,
                  kernel_size: int = 3, n_neighbors: int = 5, cache_size: int = 10000, random_state: int = 123):
        """
        Метод для создания модели машинного обучения. Все аргументы имеют значения по умолчанию, поэтому
        для каждой модели можно указывать только имеющие для нее смысл параметры.
        Типы моделей и их параметры:
        LR (линейная регрессия) - без параметров
        Ridge (LR с L2 регуляризацией) и Lasso (LR с L1 регуляризацией) - alpha (параметр регуляризации)
        ElasticNet (LR с L1 и L2 регуляризацией) - alpha (параметр регуляризации) и l1_ratio (вклад L1 относительно L2)
        SVM (метод опорных векторов) - cache_size (размер потребляемой памяти в MB) и kernel (тип ядра для kernel trick)
        GPR (гауссова регрессия) - random_state (фиксатор генератора случайных чисел)
        KNN (k-ближайших соседей) - n_neighbors (количество соседей)
        RF (случайный лес), GB (градиентный бустиен из ScickitLearn),
        XGB (градиентный бустинг из XGBoost),
        CGB (градиентный бустинг из CatBoost) - n_estimators (количество деревьев), max_depth (максимальная глубина), random_state (фиксатор генератора случайных чисел)
        FCNN (полносвязная нейронная сеть) - input_shape (размер входных данных=(кол-во признаков,)), neurons_list (список количеств нейронов для каждого скрытого слоя), loss (минимизируемая метрика), random_state (фиксатор генератора случайных чисел)
        UNet (сверточная нейронная сеть архитектуры U-Net) - conv_sample_shape (размер семпла для сверточной сети=(Ny, Nx, кол-во каналов)), n_layers - количество скрытых слоев в энкодере, n_filters - количество фильтров в первом скрытом слое, activation - функция активации, kernel_size - количество ячеек в ядре свертки
        UNet05 (сверточная нейронная сеть архитектуры, первая половина архитектуры U-Net) - conv_sample_shape (размер семпла для сверточной сети=(Ny, Nx, кол-во каналов)), n_layers - количество скрытых слоев, n_filters - количество фильтров в первом скрытом слое, activation - функция активации, kernel_size - количество ячеек в ядре свертки, max_pooling - использовать ли макс. пулинг

        """

        model_list = {
            'LR': [LinearRegression(), dict()],
            'Ridge': [Ridge, dict(alpha=alpha)],
            'Lasso': [Lasso, dict(alpha=alpha)],
            'ElasticNet': [ElasticNet, dict(alpha=alpha, l1_ratio=l1_ratio)],
            'SVM': [SVR, dict(cache_size=cache_size, kernel=kernel)],
            'GPR': [GaussianProcessRegressor, dict(random_state=random_state)],
            'KNN': [KNeighborsRegressor, dict(n_neighbors=n_neighbors)],
            'RF': [RandomForestRegressor,
                   dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)],
            'GB': [GradientBoostingRegressor,
                   dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)],
            'XGB': [XGBRegressor, dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)],
            'CGB': [CatBoostRegressor, dict(n_estimators=n_estimators, max_depth=max_depth, verbose=False,
                                            random_state=random_state)],
            'FCNN': [create_fully_connected_nn,
                     dict(input_shape=input_shape, neurons_list=neurons_list, activation=activation,
                          loss=loss, random_state=random_state)],
            'UNet': [create_unet,
                     dict(input_size=conv_sample_shape, n_layers=n_layers, n_filters=n_filters, activation=activation,
                          kernel_size=kernel_size)],
            'UNet05': [create_unet_05, dict(input_size=conv_sample_shape, n_layers=n_layers, n_filters=n_filters,
                                            activation=activation, kernel_size=kernel_size)]
        }

        if model_type not in model_list.keys():
            raise Exception('Unknown model type')
        f, params = model_list[model_type]
        self.model = f(**params)

        if model_type in ('UNet05', 'UNet'):
            self.conv_sample_shape = conv_sample_shape
        if neurons_list is None:
            neurons_list = []
        if conv_sample_shape is None:
            conv_sample_shape = [32, 32, 1]
        if metrics is None:
            metrics = ['rmse']

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.kernel = kernel
        self.neurons_list = neurons_list
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.input_shape = input_shape
        self.conv_sample_shape = conv_sample_shape
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_neighbors = n_neighbors
        self.cache_size = cache_size
        self.random_state = random_state
        tf.random.set_seed(random_state)

    def is_conv(self) -> bool:
        """
        Метод выполняет проверку, является ли модель сверточной нейронной сетью.
        Это может пригодиться при подготовки данных для обучения и предсказания.
        """

        return self.model_type in ('UNet', 'UNet05')

    def is_nn(self) -> bool:
        """
        Метод выполняет проверку, является ли модель нейронной сетью.
        Это может пригодиться при подготовки данных для обучения и предсказания.
        """

        return self.model_type in ('UNet', 'UNet05', 'FCNN')

    def evaluate(self, metric: list, y_true: ndarray, y_pred: ndarray, scaler_true: StandardScaler = None,
                 scaler_pred: StandardScaler = None) -> float:
        """
        Метод для вычисления метрики качества.
        """
        yt = y_true.copy()
        yp = y_pred.copy()

        if scaler_true is not None:
            yt = scaler_true.inverse_transform(
                y_true.reshape([-1, 1])).flatten()

        if scaler_pred is not None:
            yp = scaler_pred.inverse_transform(
                y_pred.reshape([-1, 1])).flatten()

        msk = np.logical_and(~np.isnan(yt), ~np.isnan(yp))
        yt = yt[msk]
        yp = yp[msk]

        metrics_list = {'r2': [r2_score, dict(y_true=yt, y_pred=yp)],
                        'rmse': [mean_squared_error, dict(y_true=yt, y_pred=yp, squared=False)],
                        'mpe': [mean_absolute_percentage_error, dict(y_true=yt, y_pred=yp)]
                        }
        f, param = metrics_list[metric]

        return f(**param)
        #
        # if self.metrics == 'rmse':
        #     return mean_squared_error(yt, yp, squared=False)
        # elif self.metrics == 'r2':
        #     return r2_score(yt, yp)
        # elif self.metric == 'mpe':
        #     return 100 - mean_absolute_percentage_error(yt, yp) * 100

    def calc_feature_importance(self, data: MLData, n_samples: int = None,
                                summary_plot: int = False) -> Tuple[list, np.ndarray]:
        """
        Метод для оценки важности признаков. Его можно вызывать только после обучения модели
        data - данные, на которых производится оценка
        N_samples - количество объектов, по которым производится оценка. Если None, то используются все.
        summary_plot - если True, то метод визуализирует диаграмму со значениями Шепли.
        Метод возвращает сортированный список признаков и среднее значение влияния на прогнозируемые значения.
        """

        samples = data.X
        if self.model_type == 'UNet':
            samples, _, _, _ = data.get_samples_for_encoder_decoder_CNN(self.conv_sample_shape[0],
                                                                        self.conv_sample_shape[1])
        elif self.model_type == 'UNet05':
            samples, _, _, _ = data.get_samples_for_encoder_CNN(self.conv_sample_shape[0], self.conv_sample_shape[1])

        if not n_samples is None:
            samples = samples[np.random.choice(
                samples.shape[0], n_samples, replace=False)]

        if self.is_conv():
            explainer = shap.DeepExplainer(self.model, samples)
        else:
            explainer = shap.Explainer(self.model.predict, samples)

        if self.is_conv():
            shap_values = explainer.shap_values(
                samples, check_additivity=False)
        else:
            shap_values = explainer(samples)

        if self.is_conv():
            shap_values = np.array(shap_values)
            if len(shap_values.shape) > 4:
                shap_values = shap_values[0]
            shap_values = np.sum(shap_values, axis=(1, 2))
            samples = np.mean(samples, axis=(1, 2))
        else:
            shap_values = np.array(shap_values.values)

        if data.target_processed:
            shap_values *= np.sqrt(data.target_scaler.var_)

        shap_mean = np.abs(shap_values).mean(axis=0)
        sorted_features = [name for _, name in sorted(
            zip(shap_mean, data.grid_names), reverse=True)]

        if summary_plot:
            shap.summary_plot(shap_values, samples,
                              feature_names=data.grid_names)

        shap_mean.sort()
        return sorted_features, shap_mean[::-1]

    def cross_val_score(self, data: MLData, split_type: str = 'random', n_splits: int = 5,
                        cell_size: list = None, n_epochs: int = 1,
                        print_stats: bool = False, plot_maps: bool = False, verbose: int = 2,
                        batch_size: int = 50, dx: int = 1, dy: int = 1) -> Tuple[dict, ndarray]:
        """
        Метод для кросс-валидации. Аргументы метода:
        split_type - тип разбиения. Есть три варианта: 'random' - каждая точка случайным образом
        попадает в одну из n_splits групп; 'areas' - все данные разбиваются на регионы из 
        cell_size ячеек (именно ячеек, а не метров или километров), и каждый регион случайным образом
        попадает в одну из n_splits групп; 'geographic' - вся территория разбивается на n_splits x n_splits
        участков, после чего происходит последовательный цикл из n_splits x n_splits обучений и прогнозов,
        на каждом шаге один участок откладывается для валидации, а все остальные участвуют при обучении.

        n_splits - целое число, регулирующие количество частей, на которые разбивается выборка
        metrics - список метрик
        cell_size - количество ячеек в регионах [Nx, Ny], на которые разбивается вся карта при 'areas' кросс-валидации
        n_epochs - количество эпох для обучения нейронных сетей
        print_stats - bool параметр, если True, то перед каждым циклом обучения-предсказания в консоль печатается
        статистика по трейну и тесту
        plot_maps - bool параметр, если True, то после кросс-валидации выводятся карты распределения train-test 
        на каждом шаге
        verbose - регулятор вывода нейронных сетей при обучении (см. документацию TensorFlow)
        batch_size - размер батча для обучения нейронных сетей
        dx, dy - сдвиг окна при формировании обучающей выборки для сверточных нейронных сетей        
        """
        print("---" * 30)
        print(f"Starting Cross validation with parameters: {split_type}")
        print("---" * 30)

        if cell_size is None:
            cell_size = [10, 10]

        cross_val_results = {key: list() for key in self.metrics}
        features = []
        target = []

        if split_type == 'random':
            if self.is_conv():
                raise AttributeError(
                    'Для сверточных кросс-валидация по точкам недоступна')

            kf = KFold(n_splits, shuffle=True, random_state=self.random_state)
            features = data.X
            target = data.y
            skf = kf.split(features, target)
            indices = np.arange(data.y_all.shape[0])
            indices = indices[data.get_not_nan_mask()]

        elif split_type == 'areas':
            skf = cellsKFold(data.X_all, data.y_all, n_splits, data.target_grid['data'].shape,
                             cell_size, random_state=self.random_state)
            features = data.X_all
            target = data.y_all

        elif split_type == 'geographic':
            skf = geographicKFold(data.X_all, data.y_all,
                                  n_splits, data.target_grid['data'].shape)
            features = data.X_all
            target = data.y_all
        else:
            raise AttributeError('Unknown split type')

        y_pred_all = data.y_all + np.nan
        index_map = data.y_all * 0

        step = 0
        for train_mask, test_mask in skf:
            step += 1
            print(f"Cross validation step: {step}\n")
            X_train = features[train_mask]
            y_train = target[train_mask]
            X_test = features[test_mask]
            y_test = target[test_mask]

            if print_stats:
                if data.target_processed:
                    y_train_untransf = data.target_scaler.inverse_transform(
                        y_train.reshape([-1, 1])).flatten()
                    y_test_untransf = data.target_scaler.inverse_transform(
                        y_test.reshape([-1, 1])).flatten()
                else:
                    y_train_untransf = y_train.copy()
                    y_test_untransf = y_test.copy()

                print(f'train_stats:\nShape: {y_train_untransf.shape}\nMean {np.mean(y_train_untransf)}\n'
                      f'Min: {np.min(y_train_untransf)}\nMax: {np.max(y_train_untransf)}\nSTD: {np.std(y_train_untransf)}')
                print(f'test_stats:\nShape: {y_test_untransf.shape}\nMean {np.mean(y_test_untransf)}\n'
                      f'Min: {np.min(y_test_untransf)}\nMax: {np.max(y_test_untransf)}\nSTD: {np.std(y_test_untransf)}')

            tf.keras.backend.clear_session()

            if not self.is_conv():
                if self.is_nn():
                    self.fit(X_train, y_train, n_epochs=n_epochs,
                             verbose=verbose, batch_size=batch_size)
                else:
                    self.fit(X_train, y_train)
                y_pred = self.predict(X_test, target_scaler=data.target_scaler)
            else:
                shp = data.target_grid['data'].shape
                sNx = self.conv_sample_shape[1]
                sNy = self.conv_sample_shape[0]

                y_train = data.y_all.copy()
                y_test = data.y_all.copy()

                y_train[test_mask] = np.nan
                y_test[train_mask] = np.nan

                y_train = y_train.reshape(shp)
                y_test = y_test.reshape(shp)

                Xy_train_grids = data.get_features_target_3D_grid()
                Xy_test_grids = Xy_train_grids.copy()

                Xy_train_grids[:, :, -1] = y_train
                Xy_test_grids[:, :, -1] = y_test

                X = []
                y = []
                if self.model_type == 'UNet':
                    X, indices, samples_map = get_samples_for_encoder_decoder_CNN(Xy_train_grids, sNx, sNy, dx, dy)
                    y = X[:, :, :, -1]
                    X = X[:, :, :, :-1]
                elif self.model_type == 'UNet05':
                    X, y, indices, samples_map = get_samples_for_encoder_CNN(Xy_train_grids[:, :, :-1],
                                                                             Xy_train_grids[:, :, -1],
                                                                             sNx, sNy, dx, dy)
                self.fit(X, y, n_epochs=n_epochs, verbose=verbose, batch_size=batch_size)

                if self.model_type == 'UNet':
                    X, indices_test, samples_map = get_samples_for_encoder_decoder_CNN(Xy_test_grids, sNx, sNy, 1, 1)
                    X = X[:, :, :, :-1]
                elif self.model_type == 'UNet05':
                    X, y, indices_test, samples_map = get_samples_for_encoder_CNN(Xy_test_grids[:, :, :-1],
                                                                                  Xy_test_grids[:, :, -1],
                                                                                  sNx, sNy, 1, 1)

                # with tf.device("/device:CPU:0"):
                y_pred = self.predict(X)

                if self.model_type == 'UNet':
                    x_c, y_c, y_pred = data.create_grid_by_encoder_decoder_CNN_predict(
                        y_pred, indices_test, sNx, sNy)
                else:
                    x_c, y_c, y_pred = data.create_grid_by_encoder_CNN_predict(
                        y_pred, indices_test)

                y_test = y_test.flatten()
                y_pred = y_pred.flatten()
                y_pred = data.untransform_predict(y_pred)

                y_test = y_test[test_mask]
                y_pred = y_pred[test_mask]

                # device = cuda.get_current_device()
                # device.reset()

            for metric in self.metrics:
                msk = np.logical_and(~np.isnan(y_test), ~np.isnan(y_pred))
                y_test_e = y_test[msk]
                y_pred_e = y_pred[msk]

                val = self.evaluate(metric, y_test_e, y_pred_e, scaler_true=data.target_scaler)
                cross_val_results[metric].append(val)
                print(f"Cross validation results: {metric} : {val}")

            if split_type == 'random':
                tmp_train_mask = np.zeros(data.y_all.shape[0]).astype('bool')
                tmp_test_mask = np.zeros(data.y_all.shape[0]).astype('bool')

                tmp_train_mask[indices[train_mask]] = True
                tmp_test_mask[indices[test_mask]] = True

                train_mask = tmp_train_mask
                test_mask = tmp_test_mask

            if plot_maps:
                n1 = n_splits
                if split_type == 'geographic':
                    n1 *= n1
                n2 = 1
                fig = plt.figure(figsize=(60, 30))
                ax = fig.add_subplot(n1, n2, step)
                train_test_map = data.y_all + np.nan
                train_test_map[train_mask] = 0
                train_test_map[test_mask] = 1
                ax.contourf(train_test_map.reshape(
                    data.target_grid['data'].shape))
                ax.set_aspect('equal')
                plt.axis('off')
                index_map[test_mask] = step

            y_pred_all[test_mask] = y_pred

        if plot_maps:
            fig2 = plt.figure(figsize=(20, 10))
            axi = fig2.add_subplot(111)
            mp = axi.contourf(index_map.reshape(data.target_grid['data'].shape), cmap=cm.tab20,
                              levels=np.arange(0, step + 1))
            axi.set_aspect('equal')
            plt.axis('off')
            plt.colorbar(mp, ax=axi)

        return cross_val_results, y_pred_all

    def fit(self, x_train: ndarray, y_train: ndarray, n_epochs: int = 1, batch_size: int = 50,
            validation_data: ndarray = None, verbose: int = 2, save_best_weights: bool = True):
        """
        Функция для обучения модели на заданном наборе данных.
        """

        if not self.is_nn():
            self.model.fit(x_train, y_train)
        else:
            if save_best_weights:
                if not os.path.exists('nn_fit'):
                    os.makedirs('nn_fit')

                checkpoint_filepath = 'nn_fit/'
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='loss',
                    mode='min',
                    save_best_only=True)

                self.model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
                               validation_data=validation_data, verbose=verbose, callbacks=[model_checkpoint_callback])

                self.model.load_weights(checkpoint_filepath)
            else:
                self.model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
                               validation_data=validation_data, verbose=verbose)

    def predict(self, X_test: np.ndarray, target_scaler: StandardScaler = None) -> np.ndarray:
        """
        Метод для предсказания на заданном наборе данных.
        """
        y_pred = self.model.predict(X_test)

        if self.model_type == 'UNet':
            y_pred = y_pred.reshape(y_pred.shape[:-1])

        if target_scaler is not None:
            y_pred = target_scaler.inverse_transform(
                y_pred.reshape([-1, 1])).flatten()

        return y_pred
