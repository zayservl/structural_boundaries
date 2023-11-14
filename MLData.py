import os
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ndarray

from scipy.interpolate import griddata
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

color_gradient = matplotlib.colors.LinearSegmentedColormap


def read_surfer_ascii_grid(fname: str) -> Tuple[ndarray,
                                                ndarray, ndarray]:
    """
    Функция для чтения grid-файлов формата Surfer 6 ASCII Grid.
    На вход принимает имя файла, возвращает 1D массивы координат x,y
    и 2D массив данных z
    """

    with open(fname, 'r') as f:
        if 'dsaa' in f.readline().lower():
            Nx, Ny = map(int, f.readline().split(' '))
            x_min, x_max = map(float, f.readline().split(' '))
            y_min, y_max = map(float, f.readline().split(' '))
            z_min, z_max = map(float, f.readline().split(' '))
            z = np.fromfile(f, sep=' ').reshape([Ny, Nx])
            z[z > 1.7e38] = np.nan
            x = np.linspace(x_min, x_max, Nx)
            y = np.linspace(y_min, y_max, Ny)
            return x, y, z
        else:
            raise ImportError('Unknown grid type')


def save_surfer_ascii_grid(fname: str, x: ndarray, y: ndarray, z: ndarray):
    """
    Функция для сохранения grid-файлов формата Surfer 6 ASCII Grid.
    На вход принимает имя файла, 1D массивы координат x,у и 2D массив значеинй z 
    """

    shp = z.shape
    if len(x) == shp[1] and len(y) == shp[0]:
        np.savetxt(fname, np.nan_to_num(z, nan=1.70141e38),
                   header='DSAA\n{} {}\n{} {}\n{} {}\n{} {}'.format(shp[1], shp[0], x[0], x[-1], y[0], y[-1],
                                                                    np.nanmin(z), np.nanmax(z)),
                   comments='')
    else:
        raise Exception('Shape error')


def interpolate_grid(x: ndarray, y: ndarray, grd: ndarray,
                     new_x: ndarray, new_y: ndarray, interpolation_method: str = 'nearest') -> ndarray:
    """
    Функция для 2D интерполяции. На вход принимает текущуй грид (grd) с координатами сетки x,y,
    а также новые координаты сетки (1D массивы new_x, new_y) и метод интерполяции (nearest,
    linear или cubic)

    """

    X, Y = np.meshgrid(x, y)
    X = X.reshape([-1, 1])
    Y = Y.reshape([-1, 1])
    data = grd.reshape([-1])
    if interpolation_method != 'nearest':
        X = X[~np.isnan(data)]
        Y = Y[~np.isnan(data)]
        data = data[~np.isnan(data)]

    XY = np.hstack([X, Y])
    gX, gY = np.meshgrid(new_x, new_y)

    data = griddata(XY, data, (gX, gY), method=interpolation_method)
    data[new_y > y[-1]] = np.nan
    data[new_y < y[0]] = np.nan
    data[:, new_x > x[-1]] = np.nan
    data[:, new_x < x[0]] = np.nan

    return data.reshape([len(new_y), len(new_x)])


def get_samples_for_encoder_decoder_CNN(grids: ndarray, Nx: int, Ny: int,
                                        dx: int = 1, dy: int = 1) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Вспомогательная функция для создания массива с 2D картами признаков (samples),
    и массива с 2D картами прогнозных величин в той же области.
    Размер каждой карты задается параметрами  Nx, Ny - количество ячеек по соответствующей оси.
    Параметры dx,dy регулируют сдвиг окна.
    """

    samples = []
    samples_ids = []
    samples_map = np.zeros([grids.shape[0], grids.shape[1]]) + np.nan

    i = 0
    while i < grids.shape[0] - Ny + 1:
        line_empty = True
        j = 0
        while j < grids.shape[1] - Nx + 1:
            if not np.any(np.isnan(grids[i:i + Ny, j:j + Nx, :])):
                samples.append(grids[i:i + Ny, j:j + Nx, :])
                samples_ids.append([i, j])
                samples_map[i:i + Ny, j:j + Nx] = 1
                j += dx
                line_empty = False
            else:
                j += 1

        if line_empty:
            i += 1
        else:
            i += dy

    return np.array(samples), np.array(samples_ids), samples_map


def get_samples_for_encoder_CNN(grids: ndarray, target_grid: ndarray, Nx: int, Ny: int,
                                dx: int = 1, dy: int = 1) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Вспомогательная функция для создания массива с 2D картами признаков (samples)
    и массива со значенияи прогнозных величин в центральной точке.
    Размер каждой карты задается параметрами  Nx, Ny - количество ячеек по соответствующей оси.
    Параметры dx,dy регулируют сдвиг окна.
    """

    if dx < 1 or dy < 1:
        raise Exception('Step must be greater than 1')

    samples = []
    targets = []
    samples_ids = []
    samples_map = np.zeros([grids.shape[0], grids.shape[1]])

    Nx_2 = int(Nx / 2)
    Ny_2 = int(Ny / 2)

    i = 0
    while i < grids.shape[0] - Ny + 1:
        line_empty = True
        j = 0
        while j < grids.shape[1] - Nx + 1:
            if not np.any(np.isnan(grids[i:i + Ny, j:j + Nx, :])) and not np.isnan(target_grid[i + Ny_2, j + Nx_2]):
                samples.append(grids[i:i + Ny, j:j + Nx, :])
                targets.append(target_grid[i + Ny_2, j + Nx_2])
                samples_ids.append([i + Ny_2, j + Nx_2])
                samples_map[i + Ny_2, j + Nx_2] = 1
                line_empty = False
                j += dx
            else:
                j += 1
        if line_empty:
            i += 1
        else:
            i += dy

    return np.array(samples), np.array(targets), np.array(samples_ids), samples_map


class MLData:
    """
    Класс данных для машинного обучения.
    Предназначен для импорта и хранения гридов, их интерполяции, предобработки и трансформации в таблицу признаков.
    """

    def __init__(self, interpolate_grids: bool = True, cell_size: list = None,
                 interpolation_method: str = 'nearest', regrid_mesh_type: str = 'to_target',
                 random_state: int = 123):
        """
        Конструктор. Принимает следующие аргументы:
        interpolate_grids - флаг автоматической интерполяции.
        interpolation_method - метод интерполяции на случай, если включена автоматическая интерполяция гридов.
        random_state - параметр фиксации генератора случайных чисел.
        """

        self.cell_size = [] if cell_size is None else cell_size
        self.target_grid = {}
        self.target_processed = False
        self.grids = []
        self.raw_grids = []
        self.features_processed = False
        self.interpolate = interpolate_grids
        self.interpolation_method = interpolation_method
        self.random_state = random_state
        self.target_scaler = None
        self.features_scaler = None
        self.regrid_type = regrid_mesh_type

        self.X = []
        self.y = []
        self.X_all = []
        self.y_all = []

    def import_target(self, fname: str):
        """
        Метод для импорта целевых значений для обучения из указанного файла.
        Импортированный грид сохраняется в переменную target_grid в виде словаря с ключами x, y, data.
        Также метод создает массив y_all, в котором содержатся значения таргета во всех точках (в том
        числе NaN'ы)
        """

        x, y, data = read_surfer_ascii_grid(fname)

        self.traget_processed = False
        self.x_coords = x
        self.y_coords = y
        self.target_grid = {'x': x, 'y': y, 'data': data}
        self.target_shape = data.shape
        self.y_all = data.reshape([-1])

        if self.interpolate and len(self.grids):
            self.regrid_data()

    def update_Xy(self):
        """
        Вспомогательный метод для обновления матриц признаков и таргетов для точечных прогнозов.
        """

        self.X_all = []
        self.y_all = self.target_grid['data'].flatten()

        for grid in self.grids:
            if not len(self.X_all):
                self.X_all = grid['data'].reshape([-1, 1])
            else:
                self.X_all = np.hstack(
                    [self.X_all, grid['data'].reshape([-1, 1])])

        Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])

        self.X = self.X_all[~np.isnan(Xy).any(axis=1)]
        mask = ~np.isnan(Xy).any(axis=1)
        self.X = self.X_all[mask]
        self.y = self.y_all[mask]

    def import_features(self, folder: str):
        """
        Метод для импорта признаков из нескольких файлов, расоложенных в заданной директории.
        Из заданной директории импортируются все файлы с расширением grd.
        Если при создании объекта класса MLData флаг автоматической интерполяции был выставлен True,
        то все импортируемые гриды с признаками интерполируются к сетке таргета.
        Все гриды с исходными параметрами сохраняются в список raw_grids в виде словарей с ключами x, y, и data.
        Интерполированные гриды (или исходные, если интерполяция отключена) сохраняются в список grids.
        Названия гридов (имена файлов без расширения) сохраняются в список grid_names.
        Также данный метод создает массив X_all, в котором хранятся соответствующие значения признаков во всех точках. 
        Вместе с ними создаются массивы X и y, в которых содержатся те же данные, что и в X_all и y_all,
        за исключением строк, в которых есть NaN. Массивы X и y могут использоваться для вызова метода fit() у моделей,
        которые подразумевают "точечные" прогнозы" (т.е все модели, кроме сверточных нейронных сетей).
        Данный метод необходимо вызывать после импорта таргета, чтобя для автоматической интерполяции уже была известна
        сетка таргета.
        """

        if not len(self.target_grid):
            raise ImportError('Import target first')

        self.raw_grids = []
        self.grids = []
        self.grid_names = []

        fnames = [f for f in os.listdir(folder) if f.endswith('grd')]

        for fname in fnames:
            print(fname)
            x, y, data = read_surfer_ascii_grid(folder + '/' + fname)
            self.raw_grids.append({'x': x, 'y': y, 'data': data.copy()})

            if (data.shape != self.target_grid['data'].shape or
                x[0] != self.target_grid['x'][0] or x[-1] != self.target_grid['x'][-1] or
                y[0] != self.target_grid['y'][0] or y[-1] != self.target_grid['y'][-1]) and not self.interpolate:
                raise ImportError(
                    'Different sizes of grids:\n{} and {}'.format(self.target_grid['data'].shape, data.shape))

            self.grids.append({'x': x, 'y': y, 'data': data})
            self.grid_names.append(os.path.splitext(fname)[
                                       0].replace('_', ' '))

        if self.interpolate and len(self.target_grid):
            self.regrid_data()
        else:
            self.update_Xy()
            self.features_processed = False
            Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])
            mask = ~np.isnan(Xy).any(axis=1)
            self.X = self.X_all[mask]
            self.y = self.y_all[mask]

    def regrid_data(self):
        """
        Метод для интерполяции всех загруженных данных.
        Используется после импорта признаков и таргета.
        """

        if self.regrid_type == 'to_target':
            xmin = self.target_grid['x'][0]
            xmax = self.target_grid['x'][-1]
            dx = self.target_grid['x'][1] - self.target_grid['x'][0]
            ymin = self.target_grid['y'][0]
            ymax = self.target_grid['y'][-1]
            dy = self.target_grid['y'][1] - self.target_grid['y'][0]

        # у признаков могут быть разные размеры,
        elif self.regrid_type == 'to_features' or self.regrid_type == 'min_max':
            xmin = self.raw_grids[0]['x'][0]
            xmax = self.raw_grids[0]['x'][-1]
            dx = self.raw_grids[0]['x'][1] - self.raw_grids[0]['x'][0]
            ymin = self.raw_grids[0]['y'][0]
            ymax = self.raw_grids[0]['y'][-1]
            dy = self.raw_grids[0]['y'][1] - self.raw_grids[0]['y'][0]

            for grid in self.raw_grids:
                xmin = min([xmin, grid['x'][0]])
                xmax = max([xmax, grid['x'][-1]])
                dx = min([dx, grid['x'][1] - grid['x'][0]])
                ymin = min([ymin, grid['y'][0]])
                ymax = max([ymax, grid['y'][-1]])
                dy = min([dy, grid['y'][1] - grid['y'][0]])

            if self.regrid_type == 'min_max':
                xmin = min([xmin, self.target_grid['x'][0]])
                xmax = max([xmax, self.target_grid['x'][-1]])
                dx = min([dx, self.target_grid['x'][1] - self.target_grid['x'][0]])
                ymin = min([ymin, self.target_grid['y'][0]])
                ymax = max([ymax, self.target_grid['y'][-1]])
                dy = min([dy, self.target_grid['y'][1] - self.target_grid['y'][0]])

        if len(self.cell_size) > 1:
            dx = self.cell_size[0]
            dy = self.cell_size[1]

        print('Интерполяция к сетке x: ({}, {}, {}), y: ({}, {}, {}). Тип интерполяции: {}'.format(xmin, xmax, dx,
                                                                                                   ymin, ymax, dy,
                                                                                                   self.regrid_type))

        x = np.arange(xmin, xmax + dx / 2, dx)
        y = np.arange(ymin, ymax + dy / 2, dy)

        self.x_coords = x
        self.y_coords = y

        if xmin != self.target_grid['x'][0] or xmax != self.target_grid['x'][-1] or \
                ymin != self.target_grid['y'][0] or ymax != self.target_grid['y'][-1] or \
                dx != self.target_grid['x'][1] - self.target_grid['x'][0] or \
                dy != self.target_grid['y'][1] - self.target_grid['y'][0]:
            print('Интерполяция таргета')
            data = interpolate_grid(self.target_grid['x'], self.target_grid['y'], self.target_grid['data'],
                                    x, y, self.interpolation_method)

            self.target_grid = {'x': x, 'y': y, 'data': data}
            self.target_shape = data.shape
            self.y_all = data.reshape([-1])

        for i, grid in enumerate(self.grids):
            if xmin != grid['x'][0] or xmax != grid['x'][-1] or ymin != grid['y'][0] or xmax != grid['y'][-1] or \
                    dx != grid['x'][1] - grid['x'][0] or dy != grid['y'][1] - grid['y'][0]:
                print('Интерполяция признака ' + self.grid_names[i])
                data = interpolate_grid(
                    grid['x'], grid['y'], grid['data'], x, y, self.interpolation_method)
                self.grids[i] = {'x': x, 'y': y, 'data': data}

        self.update_Xy()

    def get_not_nan_mask(self) -> ndarray:
        """
        Вспогательный метод для определения точек, в которых известны все признаки и таргет.
        """

        Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])
        return ~np.isnan(Xy).any(axis=1)

    def get_features_3D_grid(self) -> ndarray:
        """
        Вспомогательный метод для создания 3D массива из гридов с признаками. 
        Размер массива (Ny, Nx, N_grids).
        """

        shp = self.grids[0]['data'].shape
        grds = np.zeros([shp[0], shp[1], len(self.grids)])
        for i in range(len(self.grids)):
            grds[:, :, i] = self.grids[i]['data']

        return grds

    def get_features_target_3D_grid(self) -> ndarray:
        """
        Вспомогательный метод для создания 3D массива из гридов с признаками и таргетом. 
        Размер массива (Ny, Nx, N_grids+1), последний грид в данном массиве - таргет.
        """

        shp = self.grids[0]['data'].shape
        grds = np.zeros([shp[0], shp[1], len(self.grids) + 1])
        for i in range(len(self.grids)):
            grds[:, :, i] = self.grids[i]['data']
        grds[:, :, -1] = self.target_grid['data']

        return grds

    def visualize_data(self, data_type: str = 'interpolated', figsize: tuple = (15, 25),
                       ncols: int = 3, cmap: color_gradient = cm.jet, levels: int = 30):
        """
        Метод для визуализации всех загруженных карт. Аргументы методы:
        data_type - тип данных для визуализации. Есть два варианта: 'raw' - неинтерполированные карты и 
        'interpolated' - интерполированные карты.
        figsize - размер фигуры (как в matplotlib)
        ncols - количество столбцов с картами
        cmap - цветовая шкала из matplotlib
        levels - количество уровней на цветовой шкале
        """

        fig = plt.figure(figsize=figsize)
        axes = []
        Ngrids = len(self.grids)
        if len(self.target_grid):
            Ngrids += 1

        nrows = int(np.ceil(Ngrids / ncols))

        if data_type == 'interpolated':
            tmp_data = self.grids
        elif data_type == 'raw':
            tmp_data = self.raw_grids
        else:
            raise AttributeError('Unknown grids type')

        x1 = np.min([g['x'][0] for g in tmp_data])
        x2 = np.max([g['x'][-1] for g in tmp_data])
        y1 = np.min([g['y'][0] for g in tmp_data])
        y2 = np.max([g['y'][-1] for g in tmp_data])

        if len(self.target_grid):
            x1 = np.min([x1, self.x_coords[0]])
            x2 = np.max([x2, self.x_coords[-1]])
            y1 = np.min([y1, self.y_coords[0]])
            y2 = np.max([y2, self.y_coords[-1]])

        for i, name in enumerate(self.grid_names):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim([x1, x2])
            ax.set_ylim([y1, y2])
            cont = ax.contourf(
                tmp_data[i]['x'], tmp_data[i]['y'], tmp_data[i]['data'], levels=levels, cmap=cmap)
            plt.colorbar(cont, ax=ax)
            ax.set_aspect('equal')
            ax.set_title(str(i) + ' ' + name)
            plt.axis('off')

        if len(self.target_grid):
            ax = fig.add_subplot(nrows, ncols, len(self.grids) + 1)
            ax.set_xlim([x1, x2])
            ax.set_ylim([y1, y2])
            cont = ax.contourf(self.x_coords, self.y_coords,
                               self.target_grid['data'], levels=levels, cmap=cmap)
            plt.colorbar(cont, ax=ax)
            ax.set_aspect('equal')
            ax.set_title('Target')
            plt.axis('off')

    def preprocess_data(self, features_scaler: StandardScaler = None, target_scaler: StandardScaler = None):
        """
        Метод для нормализации всех признаков и таргета. В качестве аргументов можно передать уже имеющиеся скейлеры.
        Если скейлеры не переданы, то они создаются внутри метода.
        Переданные или созданные скейлеры сохраняются в переменных features_scaler и target_scaler.
        Если признаки и/или таргет уже были нормализованы, то повторная нормализация не выполняется.
        """

        if not self.features_processed:
            if features_scaler is None:
                features_scaler = StandardScaler()
                features_scaler.fit(self.X_all)

            self.X_all = features_scaler.transform(self.X_all)

            for i in range(len(self.grids)):
                self.grids[i]['data'] = self.X_all[:, i].reshape(
                    self.grids[i]['data'].shape)

            self.features_processed = True
            self.features_scaler = features_scaler

        if not self.target_processed and len(self.target_grid):
            if target_scaler is None:
                target_scaler = StandardScaler()
                target_scaler.fit(self.y_all.reshape([-1, 1]))

            self.y_all = target_scaler.transform(
                self.y_all.reshape([-1, 1])).reshape([-1])
            self.target_grid['data'] = self.y_all.reshape(
                self.target_grid['data'].shape)
            self.target_processed = True
            self.target_scaler = target_scaler

        Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])
        self.X = self.X_all[~np.isnan(Xy).any(axis=1)]
        self.y = self.y_all[~np.isnan(Xy).any(axis=1)]

    def untransform_predict(self, y_pred: ndarray) -> ndarray:
        """
        Метод для приведения прогнозов (или данных, имеющих один смысл с таргетом) к исходному масштабу.
        Может использоваться, если модель машинного обучения обучалась на нормализованном таргете. 
        Тогда и в прогнозах модели будут нормализованные значения и с помощью данного метода их можно привести
        к первоначальным масштабам, которые были до нормализации.
        """

        if self.target_processed:
            y_pred_ut = self.target_scaler.inverse_transform(
                y_pred.reshape([-1, 1])).reshape(y_pred.shape)
            return y_pred_ut
        else:
            return y_pred

    def get_samples_for_encoder_decoder_CNN(self, Nx: int, Ny: int, dx: int = 1, dy: int = 1,
                                            use_target: bool = True) -> tuple[Any, Any, ndarray, ndarray] | \
                                                                        tuple[ndarray, ndarray, ndarray]:
        """
        Метод для генерации данных для обучения (и прогноза) U-Net.
        Nx, Ny - размер окна в количестве ячеек, все карты нарезаются на семплы указанного размера.
        dx, dy - шаг окна в количестве ячеек.
        use_target - если True, то метод нарезает и возвращает не только признаки, но и таргет.
        Помимо признаков и таргета (если use_target=True), метод также возвращает массив индексов,
        который может использоваться для сбора общей прогнозной карты после предикта, а также карту
        семплов, с помощью которой можно оценить, какие области присутствуют и отсутствуют в нарезанных
        данных.
        """

        shp = self.grids[0]['data'].shape
        grds = np.zeros([shp[0], shp[1], len(self.grids) + 1])
        for i in range(len(self.grids)):
            grds[:, :, i] = self.grids[i]['data']

        if use_target:
            grds[:, :, -1] = self.target_grid['data']
            X, indices, samples_map = get_samples_for_encoder_decoder_CNN(grds, Nx, Ny, dx, dy)
            y = X[:, :, :, -1]
            X = X[:, :, :, :-1]
            return X, y, indices, samples_map

        else:
            grds = grds[:, :, :-1]
            X, indices, samples_map = get_samples_for_encoder_decoder_CNN(grds, Nx, Ny, dx, dy)
            return X, indices, samples_map

    def get_samples_for_encoder_CNN(self, Nx: int, Ny: int, dx: int = 1, dy: int = 1,
                                    use_target: bool = True) -> tuple[ndarray, ndarray, ndarray, ndarray] | \
                                                                tuple[ndarray, ndarray, ndarray]:
        """
        Метод для генерации данных для обучения (и прогноза) U-Net 1/2.
        Nx, Ny - размер окна в количестве ячеек, все карты признаков нарезаются на семплы указанного размера.
        dx, dy - шаг окна в количестве ячеек.
        use_target - если True, то метод нарезает и возвращает не только признаки, но и таргет. Значения
        таргета берутся с центральной ячейки окна.
        Помимо признаков и таргета (если use_target=True), метод также возвращает массив индексов,
        который может использоваться для сбора общей прогнозной карты после предикта, а также карту
        семплов, с помощью которой можно оценить, какие области присутствуют и отсутствуют в нарезанных
        данных.
        """

        shp = self.grids[0]['data'].shape
        grds = np.zeros([shp[0], shp[1], len(self.grids)])
        for i in range(len(self.grids)):
            grds[:, :, i] = self.grids[i]['data']

        if use_target:
            X, y, indices, samples_map = get_samples_for_encoder_CNN(grds, self.target_grid['data'], Nx, Ny, dx, dy)
            return X, y, indices, samples_map

        else:
            X, y, indices, samples_map = get_samples_for_encoder_CNN(grds, np.ones(shp), Nx, Ny, dx, dy)
            return X, indices, samples_map

    def create_grid_by_encoder_decoder_CNN_predict(self, y_pred: ndarray, indices: ndarray,
                                                   sNx: int, sNy: int) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Метод для сборки итоговой карты по прогнозам U-Net. Помимо самих прогнозов, также
        требуется передать массив индексов и размер семпла.
        """

        shp = self.grids[0]['data'].shape
        Ny = shp[0]
        Nx = shp[1]

        y_pred_map = np.zeros(shp)
        counts_map = np.zeros(shp)

        ix = 0
        iy = 0

        for i in range(y_pred.shape[0]):
            ix = indices[i, 1]
            iy = indices[i, 0]

            y_pred_map[iy:iy + sNy, ix:ix + sNx] += y_pred[i]
            counts_map[iy:iy + sNy, ix:ix + sNx] += 1

        y_pred_map[counts_map < 1] = np.nan
        y_pred_map /= counts_map

        return self.target_grid['x'], self.target_grid['y'], y_pred_map

    def create_grid_by_encoder_CNN_predict(self, y_pred: ndarray,
                                           indices: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Метод для сборки итоговой карты по прогнозам U-Net 1/2. Помимо самих прогнозов, также
        требуется передать массив индексов.
        """

        y_pred_map = np.zeros(self.target_grid['data'].shape) + np.nan

        for i in range(len(y_pred)):
            y_pred_map[indices[i, 0], indices[i, 1]] = y_pred[i]

        return self.target_grid['x'], self.target_grid['y'], y_pred_map

    def train_to_df(self, untransform: bool = False) -> pd.DataFrame:
        """
        Вспомогательный метод для создания датафрейма по обучающей выборке.
        """

        Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])
        if untransform:
            if self.features_processed:
                Xy[:, :-1] = self.features_scaler.inverse_transform(Xy[:, :-1])
            if self.target_processed:
                Xy[:, -1] = self.target_scaler.inverse_transform(
                    self.y_all.reshape([-1, 1])).flatten()

        names = self.grid_names.copy()
        names.append('target')
        return pd.DataFrame(Xy, columns=names)

    def statistics(self, untransform: bool = False) -> pd.DataFrame:
        """
        Метод для расчета статистики (минимум, максимум, среднее и т.д.) по обучающей выборке
        """

        df = self.train_to_df(untransform=untransform)
        return df.describe()

    def data_pairplot(self, n_samples: int = None, train_test_split: bool = True, kind: bool = 'hist'):
        """
        Метод для построения диаграмм рассеяния по парам признаков.
        n_samples - количество точек, которые случайным образом выбираются из выборки для 
        построения диаграмм. Если None, то берутся все точки и построение может быть долгим.
        train_test_split - если True, то точки на диаграммах раскрашиваются разными цветами
        в зависимости от наличия или отсутствия в каждой точке значения таргета.
        kind - тип диаграмм (см. sns.pairplot).
        """

        df = self.train_to_df()
        hue = None

        if n_samples is None:
            n_samples = df.shape[0]
        if (train_test_split):
            df['Is_train'] = ~np.isnan(self.y_all)
            hue = 'Is_train'

        sns.pairplot(df.sample(n_samples), dropna=True, hue=hue, kind=kind)

    def correlation_matrix(self, style: str = 'Table'):
        """
        Метод для вычисления и визуализации корреляционной матрицы.
        style - тип возвращаемого результата. Если 'Array', то возвращается датафрейм,
        если 'Table' - визуализируется раскрашенная таблица, если 'Heatmap' - визуализируется
        тепловая карта.
        """

        corr_matrix = self.train_to_df().corr()
        if style == 'Array':
            return corr_matrix
        elif style == 'Table':
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            corr_matrix[mask] = np.nan
            return corr_matrix.style.background_gradient(vmin=-1, vmax=1,
                                                         cmap='bwr').highlight_null(null_color='#f1f1f1').format(
                precision=3)
        elif style == 'Heatmap':
            return sns.heatmap(corr_matrix, cmap='bwr', vmin=-1, vmax=1)
        else:
            raise Exception('Unknown table style: ' + style)

    def remove_features_by_correlation(self, max_corr: float = 0.9):
        """
        Метод для отбраковки признаков по величине коэффициента корреляции.
        Если данный признак имеет коэффициент корреляции с другим признаком больше, чем max_corr,
        то он будет удален.
        """

        if self.features_processed:
            self.X_all = self.features_scaler.inverse_transform(self.X_all)

        corr_m = np.array(self.correlation_matrix('Array'))
        features_mask = np.ones(len(self.grids)).astype('bool')

        for i in range(len(self.grids)):
            for j in range(i):
                # если корреляция превышает пороговую и признак, с которым коррелирует текущий, не выбрасывается
                if corr_m[i, j] >= max_corr and features_mask[j]:
                    features_mask[i] = False
                    print('Удален признак {} из-за корреляции {} с признаком {}'.format(self.grid_names[i],
                                                                                        corr_m[i, j],
                                                                                        self.grid_names[j]))
                    break

        self.grids = [g for (g, m) in zip(self.grids, features_mask) if m]
        self.raw_grids = [g for (g, m) in zip(
            self.raw_grids, features_mask) if m]
        self.grid_names = [g for (g, m) in zip(
            self.grid_names, features_mask) if m]
        self.X_all = self.X_all[:, features_mask]

        if self.features_processed:
            features_scaler = StandardScaler()
            features_scaler.fit(self.X_all)
            self.X_all = features_scaler.transform(self.X_all)
            self.features_scaler = features_scaler

        Xy = np.hstack([self.X_all, self.y_all.reshape([-1, 1])])
        self.X = self.X_all[~np.isnan(Xy).any(axis=1)]
        self.y = self.y_all[~np.isnan(Xy).any(axis=1)]
