# Импортируем модуль itertools для работы с итераторами, такими как zip, cycle и т.д.
import itertools

# Импортируем модуль math для математических функций, таких как sqrt, cos, sin и radians.
import math

# Импортируем numpy как np для числовых операций, например, работы с массивами и вычисления расстояний.
import numpy as np

# Импортируем pyplot из matplotlib для создания графиков и визуализаций.
from matplotlib import pyplot as plt

# Импортируем Polygon и Rectangle из matplotlib.patches для создания многоугольников и прямоугольников.
from matplotlib.patches import Polygon, Rectangle

# Импортируем PatchCollection из matplotlib.collections для группировки нескольких фигур для эффективной отрисовки.
from matplotlib.collections import PatchCollection

# Импортируем класс Polygon из shapely.geometry как ShapelyPolygon для геометрических операций, таких как проверка пересечений.
from shapely.geometry import Polygon as ShapelyPolygon


# Определяем утилитную функцию zip_tuple для зиппинга нескольких последовательностей точек в кортежи.
def zip_tuple(*iterables):
     """Zip для нескольких последовательностей точек/вершин"""
     # Преобразуем зиппированные итераторы в список кортежей, где каждый кортеж содержит соответствующие элементы входных итераторов.
     return list(zip(*iterables))

# Определяем функцию для настройки осей графика: центрированные оси, сетка, равный масштаб и автозум.
def setup_axes(ax):
    """Спины в нуле, сетка, равный масштаб и автозум по данным."""
    # Перемещаем левую ось (y) в позицию x=0.
    ax.spines['left'].set_position('zero')
    # Перемещаем нижнюю ось (x) в позицию y=0.
    ax.spines['bottom'].set_position('zero')
    # Скрываем правую ось для более чистого вида.
    ax.spines['right'].set_color('none')
    # Скрываем верхнюю ось для более чистого вида.
    ax.spines['top'].set_color('none')
    # Устанавливаем метки x-оси внизу.
    ax.xaxis.set_ticks_position('bottom')
    # Устанавливаем метки y-оси слева.
    ax.yaxis.set_ticks_position('left')
    # Устанавливаем равный масштаб для осей x и y, чтобы сохранить пропорции фигур.
    ax.set_aspect('equal')
    # Добавляем пунктирную сетку с прозрачностью 50% для визуальной ориентации.
    ax.grid(True, linestyle=':', alpha=0.5)
    # Включаем автозум, чтобы все элементы помещались в область видимости.
    ax.autoscale(enable=True)

# Определяем функцию для визуализации последовательности многоугольников на указанных осях с заданным стилем.
def visualize_polygons(ax, polys, edgecolor='black', facecolor='none', title=None):
    # Создаем список объектов Polygon из входных многоугольников с указанными цветами контура и заливки.
    patches = [
        Polygon(p, closed=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=1.2)
        for p in polys
    ]
    # Добавляем многоугольники как PatchCollection на оси для эффективной отрисовки.
    ax.add_collection(PatchCollection(patches, match_original=True))
    # Если указан заголовок, устанавливаем его для осей с отступом 8 единиц.
    if title:
        ax.set_title(title, pad=8)
    # Применяем функцию настройки осей для единообразного стиля.
    setup_axes(ax)

# Определяем генератор прямоугольников, возвращающий координаты вершин для последовательности прямоугольников.
def gen_rectangle(width=1.0, height=0.5, spacing=0.5):
    # Инициализируем начальную x-координату для первого прямоугольника.
    x = 0.0
    # Бесконечный цикл для генерации прямоугольников.
    while True:
        # Возвращаем кортеж из четырех вершин прямоугольника, начиная с (x, 0) с заданной шириной и высотой.
        yield ((x, 0.0), (x + width, 0.0), (x + width, height), (x, height))
        # Увеличиваем x-координату на ширину прямоугольника плюс заданный отступ.
        x += width + spacing

# Определяем генератор треугольников, возвращающий координаты вершин для последовательности треугольников.
def gen_triangle(side=1.0, spacing=0.5):
    # Инициализируем начальную x-координату для первого треугольника.
    x = 0.0
    # Вычисляем высоту равностороннего треугольника по формуле h = side * sqrt(3) / 2.
    h = side * math.sqrt(3) / 2
    # Бесконечный цикл для генерации треугольников.
    while True:
        # Возвращаем кортеж из трех вершин треугольника: основание (x, 0), (x + side, 0) и вершина (x + side/2, h).
        yield ((x, 0.0), (x + side / 2, h), (x + side, 0.0))
        # Увеличиваем x-координату на длину стороны плюс заданный отступ.
        x += side + spacing

# Определяем генератор шестиугольников, возвращающий координаты вершин для последовательности шестиугольников.
def gen_hexagon(side=1.0, spacing=0.5):
    # Инициализируем начальную x-координату для первого шестиугольника.
    x = 0.0
    # Вычисляем высоту шестиугольника как h = sqrt(3) * side.
    h = math.sqrt(3) * side
    # Бесконечный цикл для генерации шестиугольников.
    while True:
        # Вычисляем x-координату центра шестиугольника.
        cx = x + side
        # Вычисляем y-координату центра шестиугольника (половина высоты).
        cy = h / 2
        # Создаем список из шести вершин шестиугольника, используя тригонометрические функции для углов 0°, 60°, 120°, и т.д.
        pts = [
            (
                cx + side * math.cos(math.radians(60 * i)),
                cy + side * math.sin(math.radians(60 * i))
            )
            for i in range(6)
        ]
        # Возвращаем кортеж вершин шестиугольника.
        yield tuple(pts)
        # Увеличиваем x-координату на удвоенную длину стороны плюс заданный отступ.
        x += 2 * side + spacing

# Определяем функцию для параллельного переноса многоугольника на заданные расстояния dx и dy.
def tr_translate(polygon, dx=0.0, dy=0.0):
    # Возвращаем новый кортеж вершин, где каждая координата (x, y) смещена на dx по x и dy по y.
    return tuple((x + dx, y + dy) for x, y in polygon)

# Определяем функцию для поворота многоугольника на заданный угол вокруг указанного центра.
def tr_rotate(polygon, angle, center=(0.0, 0.0)):
    # Преобразуем угол из градусов в радианы для использования в тригонометрических функциях.
    ang = math.radians(angle)
    # Получаем координаты центра поворота.
    cx, cy = center
    # Возвращаем новый кортеж вершин, где каждая точка повернута относительно центра по формуле поворота.
    return tuple(
        (
            cx + (x - cx) * math.cos(ang) - (y - cy) * math.sin(ang),
            cy + (x - cx) * math.sin(ang) + (y - cy) * math.cos(ang),
        )
        for x, y in polygon
    )

# Определяем функцию для симметрии многоугольника относительно указанной оси.
def tr_symmetry(polygon, axis='x'):
    # Если ось — x, отражаем y-координаты (y → -y).
    if axis == 'x':
        return tuple((x, -y) for x, y in polygon)
    # Если ось — y, отражаем x-координаты (x → -x).
    if axis == 'y':
        return tuple((-x, y) for x, y in polygon)
    # Если ось не указана, отражаем обе координаты (x → -x, y → -y).
    return tuple((-x, -y) for x, y in polygon)

# Определяем функцию для гомотетии (масштабирования) многоугольника относительно центра с коэффициентом k.
def tr_homothety(polygon, center=(0.0, 0.0), k=1.0):
    # Получаем координаты центра гомотетии.
    cx, cy = center
    # Возвращаем новый кортеж вершин, где каждая точка масштабируется относительно центра по формуле гомотетии.
    return tuple((cx + k * (x - cx), cy + k * (y - cy)) for x, y in polygon)

# Определяем функцию-фильтр для проверки, является ли многоугольник выпуклым.
def flt_convex_polygon(poly):
    # Получаем количество вершин многоугольника.
    n = len(poly)
    # Если вершин меньше 3, многоугольник не может быть выпуклым (возвращаем False).
    if n < 3:
        return False
    # Создаем список для хранения знаков векторных произведений.
    signs = []
    # Проходим по всем тройкам последовательных вершин.
    for i in range(n):
        # Получаем координаты текущей вершины.
        x1, y1 = poly[i]
        # Получаем координаты следующей вершины (с учетом замыкания).
        x2, y2 = poly[(i + 1) % n]
        # Получаем координаты следующей за следующей вершины.
        x3, y3 = poly[(i + 2) % n]
        # Вычисляем векторное произведение для определения направления поворота.
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        # Если произведение не равно нулю, добавляем его знак (True для положительного, False для отрицательного).
        if cross != 0:
            signs.append(cross > 0)
    # Многоугольник выпуклый, если все векторные произведения имеют одинаковый знак (или их нет).
    return all(signs) or not any(signs)

# Определяем функцию-фильтр, проверяющую, является ли заданная точка вершиной многоугольника.
def flt_angle_point(point):
    # Возвращаем lambda-функцию, которая проверяет, содержится ли точка в списке вершин многоугольника.
    return lambda poly: point in poly

# Определяем функцию для вычисления площади многоугольника по формуле площади через координаты вершин.
def polygon_area(poly):
    # Инициализируем переменную для хранения площади.
    area = 0
    # Получаем количество вершин многоугольника.
    n = len(poly)
    # Проходим по всем вершинам, вычисляя вклад каждой пары в площадь.
    for i in range(n):
        # Получаем координаты текущей вершины.
        x1, y1 = poly[i]
        # Получаем координаты следующей вершины (с учетом замыкания).
        x2, y2 = poly[(i + 1) % n]
        # Добавляем вклад в площадь по формуле (x1*y2 - x2*y1).
        area += x1 * y2 - x2 * y1
    # Возвращаем абсолютное значение половины вычисленной площади.
    return abs(area) / 2

# Определяем функцию-фильтр для проверки, что площадь многоугольника меньше заданного значения.
def flt_square(max_area):
    # Возвращаем lambda-функцию, которая проверяет, меньше ли площадь многоугольника заданного порога.
    return lambda poly: polygon_area(poly) < max_area

# Определяем функцию для вычисления длины кратчайшей стороны многоугольника.
def shortest_side(poly):
    # Инициализируем минимальную длину как бесконечность.
    min_len = float('inf')
    # Получаем количество вершин многоугольника.
    n = len(poly)
    # Проходим по всем сторонам многоугольника.
    for i in range(n):
        # Получаем координаты текущей вершины.
        x1, y1 = poly[i]
        # Получаем координаты следующей вершины (с учетом замыкания).
        x2, y2 = poly[(i + 1) % n]
        # Вычисляем длину стороны как евклидово расстояние между точками.
        d = math.hypot(x2 - x1, y2 - y1)
        # Обновляем минимальную длину, если текущая меньше.
        if d < min_len:
            min_len = d
    # Возвращаем длину кратчайшей стороны.
    return min_len

# Определяем функцию-фильтр для проверки, что длина кратчайшей стороны меньше заданного значения.
def flt_short_side(max_len):
    # Возвращаем lambda-функцию, которая проверяет, меньше ли длина кратчайшей стороны заданного порога.
    return lambda poly: shortest_side(poly) < max_len

# Определяем функцию для проверки, находится ли точка внутри выпуклого многоугольника.
def point_in_convex_polygon(point, polygon):
    # Получаем координаты проверяемой точки.
    x0, y0 = point
    # Получаем количество вершин многоугольника.
    n = len(polygon)
    # Инициализируем переменную для хранения предыдущего знака.
    prev = None
    # Проходим по всем сторонам многоугольника.
    for i in range(n):
        # Получаем координаты текущей вершины.
        x1, y1 = polygon[i]
        # Получаем координаты следующей вершины (с учетом замыкания).
        x2, y2 = polygon[(i + 1) % n]
        # Вычисляем векторное произведение для проверки, находится ли точка слева от стороны.
        cross = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
        # Определяем текущий знак (True, если точка слева, False, если справа).
        curr = cross >= 0
        # Если это первый знак, сохраняем его.
        if prev is None:
            prev = curr
        # Если текущий знак отличается от предыдущего, точка вне многоугольника.
        elif prev != curr:
            return False
    # Если все знаки одинаковы, точка внутри многоугольника.
    return True

# Определяем функцию-фильтр для проверки, находится ли заданная точка внутри многоугольника.
def flt_point_inside(point):
    # Возвращаем lambda-функцию, которая проверяет, находится ли точка внутри многоугольника.
    return lambda poly: point_in_convex_polygon(point, poly)

# Определяем функцию-фильтр для проверки, содержит ли многоугольник хотя бы одну вершину эталонного многоугольника.
def flt_polygon_angles_inside(ref_poly):
    # Определяем внутреннюю функцию для проверки, содержит ли многоугольник вершину из эталонного.
    def checker(poly):
        # Проходим по всем вершинам эталонного многоугольника.
        for pt in ref_poly:
            # Если вершина находится внутри многоугольника, возвращаем True.
            if point_in_convex_polygon(pt, poly):
                return True
        # Если ни одна вершина не внутри, возвращаем False.
        return False
    # Возвращаем функцию проверки.
    return checker

# Определяем декоратор для фильтрации итераторов многоугольников.
def _wrap_filter(fn):
    # Определяем декоратор, который применяет фильтр к итератору.
    def decorator(func):
        # Определяем обертку для функции.
        def wrapper(*args, **kwargs):
            # Создаем список новых аргументов.
            new_args = []
            # Проходим по всем аргументам функции.
            for arg in args:
                # Пытаемся применить фильтр к аргументу, если он итерируемый.
                try:
                    new_args.append(filter(fn, arg))
                # Если аргумент не итерируемый, добавляем его без изменений.
                except TypeError:
                    new_args.append(arg)
            # Вызываем функцию с отфильтрованными аргументами.
            return func(*new_args, **kwargs)
        # Возвращаем обернутую функцию.
        return wrapper
    # Возвращаем декоратор.
    return decorator

# Определяем декоратор для преобразования итераторов многоугольников.
def _wrap_map(fn):
    # Определяем декоратор, который применяет преобразование к итератору.
    def decorator(func):
        # Определяем обертку для функции.
        def wrapper(*args, **kwargs):
            # Создаем список новых аргументов.
            new_args = []
            # Проходим по всем аргументам функции.
            for arg in args:
                # Пытаемся применить преобразование к аргументу, если он итерируемый.
                try:
                    new_args.append(map(fn, arg))
                # Если аргумент не итерируемый, добавляем его без изменений.
                except TypeError:
                    new_args.append(arg)
            # Вызываем функцию с преобразованными аргументами.
            return func(*new_args, **kwargs)
        # Возвращаем обернутую функцию.
        return wrapper
    # Возвращаем декоратор.
    return decorator

# Создаем декоратор для фильтрации выпуклых многоугольников.
flt_convex_polygon_dec = _wrap_filter(flt_convex_polygon)

# Создаем декоратор для фильтрации многоугольников, содержащих заданную точку как вершину.
flt_angle_point_dec = lambda point: _wrap_filter(flt_angle_point(point))

# Создаем декоратор для фильтрации многоугольников с площадью меньше заданного значения.
flt_square_dec = lambda max_area: _wrap_filter(flt_square(max_area))

# Создаем декоратор для фильтрации многоугольников с кратчайшей стороной меньше заданного значения.
flt_short_side_dec = lambda max_len: _wrap_filter(flt_short_side(max_len))

# Создаем декоратор для фильтрации многоугольников, содержащих заданную точку внутри.
flt_point_inside_dec = lambda point: _wrap_filter(flt_point_inside(point))

# Создаем декоратор для фильтрации многоугольников, содержащих хотя бы одну вершину эталонного многоугольника.
flt_polygon_angles_inside_dec = lambda ref_poly: _wrap_filter(flt_polygon_angles_inside(ref_poly))

# Создаем декоратор для параллельного переноса многоугольников на dx, dy.
tr_translate_dec = lambda dx, dy: _wrap_map(lambda poly: tr_translate(poly, dx=dx, dy=dy))

# Создаем декоратор для поворота многоугольников на заданный угол.
tr_rotate_dec = lambda angle: _wrap_map(lambda poly: tr_rotate(poly, angle=angle))

# Создаем декоратор для симметрии многоугольников относительно заданной оси.
tr_symmetry_dec = lambda axis: _wrap_map(lambda poly: tr_symmetry(poly, axis=axis))

# Создаем декоратор для гомотетии многоугольников с заданным коэффициентом.
tr_homothety_dec = lambda k: _wrap_map(lambda poly: tr_homothety(poly, k=k))

# Определяем функцию для визуализации работы декораторов фильтрации и преобразования.
def visualize_decorators(sample_polygons, filters, transforms):
    # Вычисляем общее количество фильтров и преобразований.
    n = len(filters) + len(transforms)
    # Определяем количество столбцов для сетки подграфиков (минимум 2).
    cols = max(2, n // 2)
    # Вычисляем количество строк для сетки подграфиков.
    rows = (n + cols - 1) // cols
    # Создаем фигуру и массив осей для подграфиков с заданным размером.
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    # Преобразуем массив осей в одномерный для удобства.
    axes = axes.flatten()
    # Визуализируем фильтры.
    for i, (dec, title, color) in enumerate(filters):
        # Применяем декоратор фильтрации к списку многоугольников и преобразуем в список.
        decorated = list(dec(lambda x: x)(sample_polygons))
        # Получаем текущую ось для подграфика.
        ax = axes[i]
        # Визуализируем отфильтрованные многоугольники с указанным цветом.
        visualize_polygons(ax, decorated, edgecolor=color)
        # Устанавливаем заголовок для подграфика.
        ax.set_title(title)
    # Визуализируем преобразования.
    offset = len(filters)
    for j, (dec, title, color) in enumerate(transforms):
        # Применяем декоратор преобразования к списку многоугольников и преобразуем в список.
        decorated = list(dec(lambda x: x)(sample_polygons))
        # Получаем текущую ось для подграфика.
        ax = axes[offset + j]
        # Визуализируем преобразованные многоугольники с указанным цветом.
        visualize_polygons(ax, decorated, edgecolor=color)
        # Устанавливаем заголовок для подграфика.
        ax.set_title(title)
    # Удаляем пустые оси, если они есть.
    for k in range(offset + len(transforms), len(axes)):
        fig.delaxes(axes[k])
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

# Определяем функцию для визуализации пункта 4.3: параллельные ленты треугольников с симметрией.
def visualize_4_3(ax):
    # Задаем смещение по y для верхней и нижней лент треугольников.
    y_offset = 4.0
    # Задаем размер стороны треугольника.
    triangle_size = 2.0
    # Задаем шаг между треугольниками.
    step = 4
    # Проходим по x-координатам для верхних треугольников.
    for x in np.arange(-12, 12, step):
        # Создаем верхний треугольник с вершинами в (x, y_offset), (x + triangle_size, y_offset), (x + triangle_size/2, y_offset + triangle_size).
        tri_up = Polygon(
            [(x, y_offset),
             (x + triangle_size, y_offset),
             (x + triangle_size / 2, y_offset + triangle_size)],
            closed=True, edgecolor='purple', facecolor='none', linewidth=1.5)
        # Добавляем верхний треугольник на график.
        ax.add_patch(tri_up)
    # Проходим по x-координатам для нижних треугольников.
    for x in np.arange(-12, 12, step):
        # Создаем нижний треугольник с вершинами, симметричными верхнему относительно оси x.
        tri_down = Polygon(
            [(x, -y_offset),
             (x + triangle_size, -y_offset),
             (x + triangle_size / 2, -y_offset - triangle_size)],
            closed=True, edgecolor='orange', facecolor='none', linewidth=1.5)
        # Добавляем нижний треугольник на график.
        ax.add_patch(tri_down)
    # Устанавливаем пределы по x для области видимости.
    ax.set_xlim(-15, 15)
    # Устанавливаем пределы по y для области видимости.
    ax.set_ylim(-15, 15)
    # Устанавливаем заголовок для графика.
    ax.set_title("4.3 Параллельные ленты треугольников (симметрия)")
    # Применяем настройку осей для единообразного стиля.
    setup_axes(ax)

# Определяем функцию для визуализации пункта 4.4: трапеции в угле между y=x и y=-x.
def visualize_4_4(ax):
    # Задаем базовый угол для направления основания трапеций (-45°).
    base_angle = -45
    # Задаем угол роста для трапеций (45°).
    growth_angle = 45
    # Задаем начальную длину основания трапеции.
    initial_base = 0.8
    # Задаем коэффициент масштабирования для последовательных трапеций.
    scale_factor = 1.4
    # Задаем соотношение высоты трапеции к длине основания.
    height_ratio = 0.5
    # Задаем соотношение расстояния между трапециями к высоте.
    spacing_ratio = 0.3
    # Задаем количество трапеций в каждой группе.
    num_trapezoids = 6
    # Вычисляем вектор направления основания трапеции на основе базового угла.
    base_dir = np.array([np.cos(np.radians(base_angle)), np.sin(np.radians(base_angle))])
    # Создаем словарь с направлениями роста для двух квадрантов (1 и -1).
    growth_dirs = {
        1: np.array([np.cos(np.radians(growth_angle)), np.sin(np.radians(growth_angle))]),
        -1: np.array([np.cos(np.radians(growth_angle + 180)), np.sin(np.radians(growth_angle + 180))])
    }
    # Проходим по двум квадрантам (1 и -1).
    for quadrant in [1, -1]:
        # Вычисляем начальный центр трапеции для текущего квадранта.
        current_center = growth_dirs[quadrant] * initial_base * (height_ratio + spacing_ratio)
        # Проходим по количеству трапеций в каждом квадранте.
        for i in range(num_trapezoids):
            # Вычисляем длину верхнего основания трапеции (масштабируется с каждым шагом).
            upper_length = initial_base * (scale_factor ** i)
            # Вычисляем длину нижнего основания трапеции (увеличивается по сравнению с верхним).
            lower_length = upper_length * scale_factor
            # Вычисляем высоту трапеции как долю верхнего основания.
            height = upper_length * height_ratio
            # Вычисляем расстояние между трапециями как долю высоты.
            spacing = height * spacing_ratio
            # Задаем центр верхнего основания трапеции.
            upper_center = current_center
            # Вычисляем центр нижнего основания трапеции.
            lower_center = current_center + growth_dirs[quadrant] * height
            # Формируем список вершин трапеции: левая и правая точки верхнего и нижнего оснований.
            points = [
                upper_center - base_dir * upper_length / 2,
                upper_center + base_dir * upper_length / 2,
                lower_center + base_dir * lower_length / 2,
                lower_center - base_dir * lower_length / 2
            ]
            # Создаем объект Polygon для трапеции с заданным стилем.
            trapezoid = Polygon(points, closed=True, edgecolor='darkblue', facecolor='none', linewidth=1.5)
            # Добавляем трапецию на график.
            ax.add_patch(trapezoid)
            # Обновляем центр для следующей трапеции с учетом расстояния.
            current_center = lower_center + growth_dirs[quadrant] * spacing
    # Рисуем линию y=x пунктиром для визуальной ориентации.
    ax.plot([-20, 20], [20, -20], '--', color='grey', alpha=0.3)
    # Рисуем линию y=-x пунктиром для визуальной ориентации.
    ax.plot([-20, 20], [-20, 20], '--', color='grey', alpha=0.3)
    # Устанавливаем пределы по x для области видимости.
    ax.set_xlim(-15, 15)
    # Устанавливаем пределы по y для области видимости.
    ax.set_ylim(-15, 15)
    # Устанавливаем заголовок для графика.
    ax.set_title("4.4 Трапеции в угле между y=x и y=-x")
    # Применяем настройку осей для единообразного стиля.
    setup_axes(ax)

# Определяем функцию для генерации трапеций из пункта 4.4.
def generate_trapezoids_4_4(num_trapezoids=6):
    # Задаем базовый угол для направления основания трапеций (-45°).
    base_angle = -45
    # Задаем угол роста для трапеций (45°).
    growth_angle = 45
    # Задаем начальную длину основания трапеции.
    initial_base = 0.8
    # Задаем коэффициент масштабирования для последовательных трапеций.
    scale_factor = 1.4
    # Задаем соотношение высоты трапеции к длине основания.
    height_ratio = 0.5
    # Задаем соотношение расстояния между трапециями к высоте.
    spacing_ratio = 0.3
    # Вычисляем вектор направления основания трапеции.
    base_dir = np.array([np.cos(np.radians(base_angle)), np.sin(np.radians(base_angle))])
    # Создаем словарь с направлениями роста для двух квадрантов.
    growth_dirs = {
        1: np.array([np.cos(np.radians(growth_angle)), np.sin(np.radians(growth_angle))]),
        -1: np.array([np.cos(np.radians(growth_angle + 180)), np.sin(np.radians(growth_angle + 180))])
    }
    # Создаем пустой список для хранения трапеций.
    trapezoids = []
    # Проходим по двум квадрантам (1 и -1).
    for quadrant in [1, -1]:
        # Вычисляем начальный центр трапеции для текущего квадранта.
        current_center = growth_dirs[quadrant] * initial_base * (height_ratio + spacing_ratio)
        # Проходим по количеству трапеций в каждом квадранте.
        for i in range(num_trapezoids):
            # Вычисляем длину верхнего основания трапеции.
            upper_length = initial_base * (scale_factor ** i)
            # Вычисляем длину нижнего основания трапеции.
            lower_length = upper_length * scale_factor
            # Вычисляем высоту трапеции.
            height = upper_length * height_ratio
            # Вычисляем расстояние между трапециями.
            spacing = height * spacing_ratio
            # Задаем центр верхнего основания трапеции.
            upper_center = current_center
            # Вычисляем центр нижнего основания трапеции.
            lower_center = current_center + growth_dirs[quadrant] * height
            # Формируем список вершин трапеции.
            points = [
                upper_center - base_dir * upper_length / 2,
                upper_center + base_dir * upper_length / 2,
                lower_center + base_dir * lower_length / 2,
                lower_center - base_dir * lower_length / 2
            ]
            # Добавляем трапецию (кортеж вершин) в список.
            trapezoids.append(tuple(points))
            # Обновляем центр для следующей трапеции.
            current_center = lower_center + growth_dirs[quadrant] * spacing
    # Возвращаем список трапеций.
    return trapezoids

# Определяем функцию для генерации масштабированных и повернутых трапеций (пункт 6.1).
def generate_scaled_rotated_squares():
    # Возвращаем итератор трапеций из функции generate_trapezoids_4_4 с 6 трапециями.
    return iter(generate_trapezoids_4_4(num_trapezoids=6))

# Определяем функцию для фильтрации первых 6 фигур из итератора.
def filter_6_figures(polygons):
    # Возвращаем список из первых 6 многоугольников, используя itertools.islice.
    return list(itertools.islice(polygons, 6))

# Определяем функцию для генерации 15 фигур с разным масштабом (пункт 6.2).
def generate_mixed_scaled_figures():
    # Создаем список генераторов для прямоугольников, треугольников и шестиугольников.
    gens = [
        gen_rectangle(width=1, height=0.5, spacing=0.5),
        gen_triangle(side=1, spacing=0.5),
        gen_hexagon(side=1, spacing=0.5),
    ]
    # Создаем циклический итератор для чередования генераторов.
    cycl = itertools.cycle(gens)
    # Создаем пустой список для хранения фигур.
    figures = []
    # Создаем циклический итератор для масштабов (0.5, 1, 1.5, 2, 2.5).
    scales = itertools.cycle([0.5, 1, 1.5, 2, 2.5])
    # Генерируем 15 фигур.
    for _ in range(15):
        # Получаем следующую фигуру из текущего генератора.
        poly = next(next(cycl))
        # Получаем следующий коэффициент масштабирования.
        k = next(scales)
        # Применяем гомотетию с текущим коэффициентом масштаба.
        scaled = tr_homothety(poly, k=k)
        # Добавляем масштабированную фигуру в список.
        figures.append(scaled)
    # Возвращаем список масштабированных фигур.
    return figures

# Определяем функцию для фильтрации многоугольников по длине кратчайшей стороны.
def filter_short_side(polygons, max_length):
    # Фильтруем многоугольники, у которых длина кратчайшей стороны меньше max_length.
    return list(filter(flt_short_side(max_length), polygons))

# Определяем функцию для проверки пересечения двух многоугольников с использованием Shapely.
def polygons_intersect(poly1, poly2):
    # Создаем объект ShapelyPolygon для первого многоугольника.
    sp1 = ShapelyPolygon(poly1)
    # Создаем объект ShapelyPolygon для второго многоугольника.
    sp2 = ShapelyPolygon(poly2)
    # Проверяем, пересекаются ли два многоугольника, и возвращаем результат.
    return sp1.intersects(sp2)

# Определяем функцию для фильтрации непересекающихся многоугольников.
def filter_non_intersecting(polygons):
    # Создаем пустой список для хранения непересекающихся многоугольников.
    result = []
    # Проходим по каждому многоугольнику.
    for poly in polygons:
        # Проверяем, не пересекается ли текущий многоугольник с уже добавленными.
        if not any(polygons_intersect(poly, p) for p in result):
            # Если нет пересечений, добавляем многоугольник в результат.
            result.append(poly)
    # Возвращаем список непересекающихся многоугольников.
    return result

# Определяем функцию для визуализации многоугольников с заголовком и заданным стилем.
def visualize_polygons_with_title(polygons, title, edgecolor='blue', figsize=(8, 8)):
    # Создаем фигуру и ось с заданным размером.
    fig, ax = plt.subplots(figsize=figsize)
    # Проходим по каждому многоугольнику.
    for poly in polygons:
        # Создаем объект Polygon для текущего многоугольника с заданным стилем.
        patch = Polygon(poly, closed=True, edgecolor=edgecolor, facecolor='none', linewidth=2)
        # Добавляем многоугольник на график.
        ax.add_patch(patch)
    # Применяем настройку осей для единообразного стиля.
    setup_axes(ax)
    # Устанавливаем заголовок графика.
    ax.set_title(title, fontsize=14)
    # Отображаем фигуру.
    plt.show()

# Определяем функцию для визуализации преобразований многоугольников (пункт 3).
def visualize_operations_on_figures():
    # Создаем фигуру и массив осей 2x2 с заданным размером.
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # Преобразуем массив осей в одномерный для удобства.
    axs = axs.flatten()
    # Генерируем 5 прямоугольников с заданными параметрами.
    original_squares = list(itertools.islice(gen_rectangle(width=1.5, height=1, spacing=1), 5))
    # Применяем параллельный перенос (dx=2, dy=2) к прямоугольникам.
    translated = list(map(lambda p: tr_translate(p, dx=2, dy=2), original_squares))
    # Визуализируем перенесенные прямоугольники с красным контуром.
    visualize_polygons(axs[0], translated, edgecolor='red')
    # Устанавливаем заголовок для подграфика.
    axs[0].set_title("3.1: Параллельный перенос (dx=2, dy=2)")
    # Применяем поворот на 45° к прямоугольникам.
    rotated = list(map(lambda p: tr_rotate(p, angle=45), original_squares))
    # Визуализируем повернутые прямоугольники с зеленым контуром.
    visualize_polygons(axs[1], rotated, edgecolor='green')
    # Устанавливаем заголовок для подграфика.
    axs[1].set_title("3.2: Поворот на 45°")
    # Применяем симметрию относительно оси Y к прямоугольникам.
    symmetric = list(map(lambda p: tr_symmetry(p, axis='y'), original_squares))
    # Визуализируем симметричные прямоугольники с синим контуром.
    visualize_polygons(axs[2], symmetric, edgecolor='blue')
    # Устанавливаем заголовок для подграфика.
    axs[2].set_title("3.3: Симметрия относительно Y")
    # Применяем гомотетию с коэффициентом 1.5 к прямоугольникам.
    homothetic = list(map(lambda p: tr_homothety(p, k=1.5), original_squares))
    # Визуализируем масштабированные прямоугольники с фиолетовым контуром.
    visualize_polygons(axs[3], homothetic, edgecolor='purple')
    # Устанавливаем заголовок для подграфика.
    axs[3].set_title("3.4: Гомотетия (k=1.5)")
    # Применяем настройку осей для всех подграфиков.
    for ax in axs:
        setup_axes(ax)
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

# Определяем функцию для визуализации пункта 7.1: фильтрация по площади.
def visualize_7_1(ax):
    # Задаем порог площади для фильтрации.
    threshold = 4.0
    # Определяем список примеров фигур: два треугольника и один прямоугольник.
    samples = [
        [(0,0),(2,0),(1,3)],
        [(4,0),(6,0),(6,2),(4,2)],
        [(7,2),(9,2),(8,5)]
    ]
    # Фильтруем фигуры, у которых площадь меньше порога.
    accepted = list(filter(flt_square(threshold), samples))
    # Создаем список фигур, которые не прошли фильтрацию.
    rejected = [p for p in samples if p not in accepted]
    # Рисуем отклоненные фигуры розовым пунктиром.
    for poly in rejected:
        patch = Polygon(poly, closed=True,
                        edgecolor='hotpink', facecolor='none',
                        linestyle='--', linewidth=3)
        ax.add_patch(patch)
    # Рисуем принятые фигуры красным и подписываем их площадь.
    for poly in accepted:
        patch = Polygon(poly, closed=True,
                        edgecolor='red', facecolor='none', linewidth=3)
        ax.add_patch(patch)
        # Вычисляем центр многоугольника для размещения текста.
        cx = sum(x for x,y in poly)/len(poly)
        cy = sum(y for x,y in poly)/len(poly)
        # Добавляем текст с площадью в центре многоугольника.
        ax.text(cx, cy, f"{polygon_area(poly):.1f}",
                color='red', fontsize=14,
                ha='center', va='center')
    # Применяем настройку осей.
    setup_axes(ax)
    # Устанавливаем заголовок с описанием фильтрации.
    ax.set_title(f"7.1: Фильтрация (красные) и отклоненные (розовые). фигуры с площадью < {threshold:.1f}.")

# Определяем функцию для визуализации пункта 7.2: преобразование фигур.
def visualize_7_2(ax):
    # Генерируем 4 треугольника с заданными параметрами.
    samples = list(itertools.islice(gen_triangle(side=2, spacing=3), 4))
    # Рисуем оригинальные треугольники серым пунктиром.
    for poly in samples:
        patch = Polygon(poly, closed=True,
                        edgecolor='lightgray', facecolor='none',
                        linestyle='--')
        ax.add_patch(patch)
    # Применяем поворот на 30° и сдвиг на (2, 1) к треугольникам.
    for poly in samples:
        t = tr_translate(tr_rotate(poly, 30), 2, 1)
        # Рисуем преобразованные треугольники зеленым.
        patch = Polygon(t, closed=True,
                        edgecolor='green', facecolor='none', linewidth=3)
        ax.add_patch(patch)
        # Вычисляем центр преобразованного треугольника.
        cx = sum(x for x,y in t)/len(t)
        cy = sum(y for x,y in t)/len(t)
        # Добавляем текст с площадью (2.0) в центре треугольника.
        ax.text(cx, cy, "2.0",
                color='green', fontsize=14,
                ha='center', va='center')
    # Применяем настройку осей.
    setup_axes(ax)
    # Устанавливаем заголовок с описанием преобразования.
    ax.set_title("7.2: Преобразованные фигуры. поворот на 30°, затем сдвиг на (2, 1).")

# Определяем функцию для аналитической визуализации многоугольников (пункт 8).
def visualize_8(ax):
    # Импортируем reduce из functools для агрегации данных.
    from functools import reduce
    # Определяем набор из пяти многоугольников.
    polys = [
        [(0,0),(2,0),(1,2)],
        [(3,1),(5,1),(4,3)],
        [(1,1),(2,2),(1,3),(0,2)],
        [(6,2),(8,2),(7,4)],
        [(0,0),(4,0),(4,3),(0,3)]
    ]
    # Определяем функцию-агрегатор для поиска ближайшей точки к началу координат.
    def agr_origin_nearest(acc, poly):
        import numpy as _np
        # Находим вершину с минимальным расстоянием до (0,0).
        min_pt = min(poly, key=lambda p: _np.hypot(p[0], p[1]))
        # Возвращаем точку с меньшим расстоянием от начального аккумулятора.
        return min_pt if _np.hypot(min_pt[0], min_pt[1]) < _np.hypot(acc[0], acc[1]) else acc
    # Определяем функцию-агрегатор для поиска самой длинной стороны.
    def agr_max_side(acc, poly):
        import numpy as _np
        # Вычисляем длину каждой стороны и находим максимальную.
        ms = max(_np.hypot(poly[i][0]-poly[(i+1)%len(poly)][0], poly[i][1]-poly[(i+1)%len(poly)][1]) for i in range(len(poly)))
        # Возвращаем максимальную сторону, если она больше аккумулятора.
        return ms if acc is None or ms>acc else acc
    # Определяем функцию-агрегатор для поиска минимальной площади.
    def agr_min_area(acc, poly):
        # Вычисляем площадь многоугольника.
        ar = polygon_area(poly)
        # Возвращаем минимальную площадь.
        return ar if acc is None or ar<acc else acc
    # Определяем функцию-агрегатор для суммирования периметров.
    def agr_perimeter(acc, poly):
        import numpy as _np
        # Вычисляем периметр многоугольника как сумму длин сторон.
        per = sum(_np.hypot(poly[i][0]-poly[(i+1)%len(poly)][0], poly[i][1]-poly[(i+1)%len(poly)][1]) for i in range(len(poly)))
        # Добавляем периметр к аккумулятору.
        return acc+per
    # Определяем функцию-агрегатор для суммирования площадей.
    def agr_area(acc, poly):
        # Добавляем площадь многоугольника к аккумулятору.
        return acc+polygon_area(poly)
    # Импортируем array и inf из numpy для вычислений.
    from numpy import array, inf
    # Находим ближайшую точку к началу координат с помощью reduce.
    closest = reduce(lambda a,p: agr_origin_nearest(a,p), polys, [inf,inf])
    # Находим самую длинную сторону с помощью reduce.
    max_side = reduce(lambda a,p: agr_max_side(a,p), polys, None)
    # Находим минимальную площадь с помощью reduce.
    min_area = reduce(lambda a,p: agr_min_area(a,p), polys, None)
    # Вычисляем суммарный периметр с помощью reduce.
    total_per = reduce(lambda a,p: agr_perimeter(a,p), polys, 0.0)
    # Вычисляем суммарную площадь с помощью reduce.
    total_area = reduce(lambda a,p: agr_area(a,p), polys, 0.0)
    # Определяем цвета для каждого многоугольника.
    colors = ['blue','orange','green','red','purple']
    # Создаем метки для каждого многоугольника.
    labels = [f'Многоугольник {i+1}' for i in range(len(polys))]
    # Рисуем каждый многоугольник с соответствующим цветом и меткой.
    for poly,c,lbl in zip(polys,colors,labels):
        ax.add_patch(Polygon(poly, closed=True, edgecolor=c, facecolor='none', linewidth=2, label=lbl))
    # Рисуем ближайшую точку к началу координат желтым маркером.
    ax.scatter(closest[0], closest[1], s=100, edgecolor='black', facecolor='yellow', zorder=5,
               label=f'Ближайшая точка: ({closest[0]:.2f}, {closest[1]:.2f})')
    # Создаем список статистических данных для отображения.
    stats=[
        f'1. Ближайшая точка к началу: ({closest[0]:.2f}, {closest[1]:.2f})',
        f'2. Самая длинная сторона: {max_side:.2f}',
        f'3. Минимальная площадь: {min_area:.2f}',
        f'4. Суммарный периметр: {total_per:.2f}',
        f'5. Суммарная площадь: {total_area:.2f}',
    ]
    # Добавляем статистику на график в верхнем левом углу.
    for i,t in enumerate(stats):
        ax.text(0.02,0.95-i*0.05,t,transform=ax.transAxes,fontsize=12,va='top')
    # Применяем настройку осей.
    setup_axes(ax)
    # Устанавливаем пределы по x для области видимости.
    ax.set_xlim(-2,10)
    # Устанавливаем пределы по y для области видимости.
    ax.set_ylim(-2,5)
    # Устанавливаем заголовок графика.
    ax.set_title('Пункт 8: Аналитическая визуализация многоугольников', pad=20)
    # Добавляем легенду в правый нижний угол.
    ax.legend(loc='lower right')

# Проверяем, запущен ли скрипт напрямую.
if __name__ == '__main__':
    # Создаем фигуру и массив осей 2x2 для пункта 2.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Генерируем 7 прямоугольников.
    rects = list(itertools.islice(gen_rectangle(width=2, height=1, spacing=1), 7))
    # Визуализируем прямоугольники с красным контуром.
    visualize_polygons(axs[0, 0], rects, edgecolor='red', title='2.1 Прямоугольники')
    # Генерируем 7 треугольников.
    tris = list(itertools.islice(gen_triangle(side=2, spacing=1), 7))
    # Визуализируем треугольники с зеленым контуром.
    visualize_polygons(axs[0, 1], tris, edgecolor='green', title='2.2 Треугольники')
    # Генерируем 7 шестиугольников.
    hexs = list(itertools.islice(gen_hexagon(side=1.5, spacing=1), 7))
    # Визуализируем шестиугольники с синим контуром.
    visualize_polygons(axs[1, 0], hexs, edgecolor='blue', title='2.3 Шестиугольники')
    # Создаем список для смешанных фигур.
    mixed = []
    # Создаем список генераторов для прямоугольников, треугольников и шестиугольников.
    gens = [gen_rectangle(), gen_triangle(), gen_hexagon()]
    # Создаем циклический итератор для чередования генераторов.
    cycl = itertools.cycle(gens)
    # Генерируем 7 смешанных фигур.
    for _ in range(7):
        mixed.append(next(next(cycl)))
    # Визуализируем смешанные фигуры с фиолетовым контуром.
    visualize_polygons(axs[1, 1], mixed, edgecolor='purple', title='2.4 Смешанная')
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # От displaying the figure.
    plt.show()

    # Создаем фигуру и массив осей 2x2 для пункта 4.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Получаем первую ось для пункта 4.1.
    ax = axs[0, 0]
    # Создаем пустой список для хранения прямоугольников.
    patches = []
    # Проходим по трем значениям смещения по y.
    for dy in (0, 2, -2):
        # Генерируем 7 прямоугольников с поворотом на 30° и смещением по y.
        seq = itertools.islice(
            (tr_translate(tr_rotate(p, 30), dy=dy) for p in gen_rectangle(width=1, height=0.5, spacing=1)),
            7,
        )
        # Создаем объект Polygon для каждого прямоугольника.
        for p in seq:
            patches.append(Polygon(p, closed=True, edgecolor='navy', facecolor='none', linewidth=1.2))
    # Добавляем прямоугольники как PatchCollection на ось.
    ax.add_collection(PatchCollection(patches, match_original=True))
    # Устанавливаем заголовок для подграфика.
    ax.set_title('4.1 Параллельные ленты')
    # Применяем настройку осей.
    setup_axes(ax)
    # Получаем вторую ось для пункта 4.2.
    ax = axs[0, 1]
    # Задаем углы для двух лент прямоугольников.
    angle1, angle2 = 30, 150
    # Задаем смещение для лент.
    shift = 5
    # Задаем ширину и высоту прямоугольников.
    rect_w, rect_h = 2.5, 0.8
    # Проходим по параметру t для создания прямоугольников.
    for t in np.arange(-10, 10, 2.5):
        # Вычисляем координаты центра первого прямоугольника.
        dx1 = t * math.cos(math.radians(angle1)) + shift
        dy1 = t * math.sin(math.radians(angle1)) + shift
        # Создаем первый прямоугольник с поворотом на angle1.
        rect1 = Rectangle(
            (dx1 - rect_w / 2, dy1 - rect_h / 2), rect_w, rect_h, angle=angle1, edgecolor='red', facecolor='none', linewidth=1.2
        )
        # Добавляем первый прямоугольник на график.
        ax.add_patch(rect1)
        # Вычисляем координаты центра второго прямоугольника.
        dx2 = t * math.cos(math.radians(angle2)) - shift
        dy2 = t * math.sin(math.radians(angle2)) + shift
        # Создаем второй прямоугольник с поворотом на angle2.
        rect2 = Rectangle(
            (dx2 - rect_w / 2, dy2 - rect_h / 2), rect_w, rect_h, angle=angle2, edgecolor='green', facecolor='none', linewidth=1.2
        )
        # Добавляем второй прямоугольник на график.
        ax.add_patch(rect2)
    # Устанавливаем заголовок для подграфика.
    ax.set_title('4.2 Пересекающиеся ленты')
    # Применяем настройку осей.
    setup_axes(ax)
    # Устанавливаем пределы по x.
    ax.set_xlim(-15, 15)
    # Устанавливаем пределы по y.
    ax.set_ylim(-10, 15)
    # Визуализируем пункт 4.3 на третьей оси.
    visualize_4_3(axs[1, 0])
    # Визуализируем пункт 4.4 на четвертой оси.
    visualize_4_4(axs[1, 1])
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

    # Определяем список многоугольников для пункта 5.
    polygons = [
        [(0, 0), (3, 0), (3, 3), (0, 3)],
        [(3, 3), (4, 3), (4, 4), (3, 4)],
        [(1, 1), (2, 1), (2, 2), (1, 2)],
        [(7, 7), (9, 7), (8, 9)],
        [(5, 5), (6, 6), (7, 5), (6, 4)],
        [(10, 10), (12, 11), (11, 13), (9, 12)],
    ]
    # Задаем тестовую точку для фильтрации по вершинам.
    test_point_angle = (5, 5)
    # Задаем тестовую точку для фильтрации по нахождению внутри.
    test_point_inside = (5.5, 5.5)
    # Задаем эталонный многоугольник для фильтрации.
    ref_polygon = polygons[0]
    # Определяем список фильтров с их заголовками и цветами.
    filters = [
        (flt_convex_polygon, "5.1: Выпуклые многоугольники", "blue"),
        (flt_angle_point(test_point_angle), f"5.2: Угол в {test_point_angle}", "green"),
        (flt_square(1.5), "5.3: Площадь < 1.5", "red"),
        (flt_short_side(1.5), "5.4: Кратчайшая сторона < 1.5", "purple"),
        (flt_point_inside(test_point_inside), f"5.5: Точка {test_point_inside} внутри", "orange"),
        (flt_polygon_angles_inside(ref_polygon), "5.6: Содержит углы первого полигона", "brown"),
    ]
    # Создаем фигуру и массив осей 2x3 для пункта 5.
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # Преобразуем массив осей в одномерный.
    axs = axs.flatten()
    # Проходим по каждому фильтру и соответствующей оси.
    for ax, (filt, title, color) in zip(axs, filters):
        # Фильтруем многоугольники с использованием текущего фильтра.
        filtered = list(filter(filt, polygons))
        # Рисуем все многоугольники серым пунктиром (исходные).
        for poly in polygons:
            patch = Polygon(poly, closed=True, edgecolor='lightgray', facecolor='none', linestyle='--', linewidth=1)
            ax.add_patch(patch)
        # Рисуем отфильтрованные многоугольники с указанным цветом.
        for poly in filtered:
            patch = Polygon(poly, closed=True, edgecolor=color, facecolor='none', linewidth=3)
            ax.add_patch(patch)
        # Применяем настройку осей.
        setup_axes(ax)
        # Устанавливаем заголовок подграфика.
        ax.set_title(title, fontsize=14)
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

    # Генерируем все трапеции для пункта 6.1.
    all_traps = generate_trapezoids_4_4(num_trapezoids=6)
    # Выбираем первые 6 трапеций (положительный квадрант).
    pos_traps = all_traps[:6]
    # Выбираем следующие 6 трапеций (отрицательный квадрант).
    neg_traps = all_traps[6:12]
    # Создаем фигуру и ось для пункта 6.1.
    fig6, ax6 = plt.subplots(figsize=(8, 8))
    # Рисуем трапеции положительного квадранта сплошной синей линией.
    for poly in pos_traps:
        patch = Polygon(poly, closed=True,
                        edgecolor='darkblue', facecolor='none', linewidth=2)
        ax6.add_patch(patch)
    # Рисуем трапеции отрицательного квадранта розовым пунктиром.
    for poly in neg_traps:
        patch = Polygon(poly, closed=True,
                        edgecolor='hotpink', facecolor='none',
                        linestyle='--', linewidth=2, alpha=0.7)
        ax6.add_patch(patch)
    # Применяем настройку осей.
    setup_axes(ax6)
    # Устанавливаем заголовок графика.
    ax6.set_title("6.1: Фильтрация 6 фигур из 4.4", fontsize=14)
    # Отображаем фигуру.
    plt.show()

    # Генерируем 15 масштабированных фигур для пункта 6.2.
    mixed_scaled = generate_mixed_scaled_figures()
    # Фильтруем фигуры с кратчайшей стороной меньше 0.6.
    filtered_short = filter_short_side(mixed_scaled, max_length=0.6)
    # Визуализируем все масштабированные фигуры серым цветом.
    visualize_polygons_with_title(mixed_scaled, "6.2: 15 фигур с разным масштабом", edgecolor='gray')
    # Визуализируем отфильтрованные фигуры фиолетовым цветом.
    visualize_polygons_with_title(filtered_short, "6.2: Фильтрация по кратчайшей стороне < 0.6", edgecolor='purple')

    # Фильтруем непересекающиеся фигуры для пункта 6.3.
    filtered_no_intersections = filter_non_intersecting(mixed_scaled)
    # Визуализируем исходные фигуры серым цветом.
    visualize_polygons_with_title(mixed_scaled, "6.3: Исходные фигуры (с пересечениями)", edgecolor='gray')
    # Визуализируем непересекающиеся фигуры зеленым цветом.
    visualize_polygons_with_title(filtered_no_intersections, "6.3: Фильтрация непересекающихся фигур", edgecolor='green')

    # Вызываем визуализацию преобразований для пункта 3.
    visualize_operations_on_figures()

    # Определяем список многоугольников для пункта 7.
    sample_polygons = [
        [(0,0),(2,0),(1,2)],
        [(0,0),(1,0),(1,1),(0,1)],
        [(2,2),(3,2),(3,3)]
    ]
    # Определяем список фильтров для визуализации декораторов.
    filters = [
        (flt_convex_polygon_dec, '7.1: flt_convex_polygon', 'blue'),
        (flt_angle_point_dec((0,0)), '7.2: flt_angle_point (0,0)', 'green'),
        (flt_square_dec(2.0), '7.3: flt_square (<2.0)', 'red'),
        (flt_short_side_dec(1.5), '7.4: flt_short_side (<1.5)', 'purple'),
        (flt_point_inside_dec((0.5,0.5)), '7.5: flt_point_inside (0.5,0.5)', 'orange'),
        (flt_polygon_angles_inside_dec([(0,0),(1,0),(1,1),(0,1)]), '7.6: flt_polygon_angles_inside', 'brown')
    ]
    # Определяем список преобразований для визуализации декораторов.
    transforms = [
        (tr_translate_dec(1.0,1.0), '7.7: tr_translate (+1,+1)', 'cyan'),
        (tr_rotate_dec(90), '7.8: tr_rotate (90°)', 'magenta'),
        (tr_symmetry_dec('y'), '7.9: tr_symmetry (y)', 'black'),
        (tr_homothety_dec(2.0), '7.10: tr_homothety (k=2.0)', 'gray')
    ]
    # Вызываем визуализацию декораторов.
    visualize_decorators(sample_polygons, filters, transforms)

    # Создаем фигуру и массив осей 1x2 для пункта 7.
    fig7, axes7 = plt.subplots(1, 2, figsize=(16, 6))
    # Визуализируем пункт 7.1 на первой оси.
    visualize_7_1(axes7[0])
    # Визуализируем пункт 7.2 на второй оси.
    visualize_7_2(axes7[1])
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

    # Создаем фигуру и ось для пункта 8.
    fig8, ax8 = plt.subplots(figsize=(12,6))
    # Визуализируем пункт 8.
    visualize_8(ax8)
    # Оптимизируем расположение графика.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

    # Определяем функцию для склейки треугольников в шестиугольники (пункт 9.1).
    def zip_polygons(top_polygons, bottom_polygons):
        """Склейка треугольников в шестиугольники"""
        # Создаем шестиугольники, объединяя вершины верхнего треугольника с перевернутыми вершинами нижнего.
        return [list(top) + list(bottom[::-1])
                for top, bottom in zip(top_polygons, bottom_polygons)]
    # Определяем вершины для верхних треугольников.
    top_vertices   = [(1,1),(2,2),(3,1), (4,1),(5,2),(6,1), (7,1),(8,2),(9,1), (10,1),(11,2),(12,1)]
    # Создаем вершины для нижних треугольников, симметричных верхним относительно оси x.
    bottom_vertices= [(x,-y) for (x,y) in top_vertices]
    # Разбиваем верхние вершины на треугольники, группируя по три.
    top_triangles    = zip_tuple(top_vertices[::3], top_vertices[1::3], top_vertices[2::3])
    # Разбиваем нижние вершины на треугольники, группируя по три.
    bottom_triangles = zip_tuple(bottom_vertices[::3], bottom_vertices[1::3], bottom_vertices[2::3])
    # Склеиваем треугольники в шестиугольники.
    hexagons = zip_polygons(top_triangles, bottom_triangles)
    # Создаем фигуру и три оси для пункта 9.1.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
    # Рисуем верхние и нижние вершины как точки на первой оси.
    ax1.scatter(*zip(*top_vertices), color='blue')
    ax1.scatter(*zip(*bottom_vertices), color='blue')
    # Применяем настройку осей.
    setup_axes(ax1)
    # Устанавливаем заголовок для подграфика.
    ax1.set_title('9.1. Вершины (zip_polygons)')
    # Рисуем верхние треугольники зеленым на второй оси.
    for tri in top_triangles:
        ax2.add_patch(Polygon(tri, closed=True, ec='green', lw=2))
    # Рисуем нижние треугольники красным на второй оси.
    for tri in bottom_triangles:
        ax2.add_patch(Polygon(tri, closed=True, ec='red', lw=2))
    # Применяем настройку осей.
    setup_axes(ax2)
    # Устанавливаем заголовок для подграфика.
    ax2.set_title('9.1. Треугольники (zip_tuple)')
    # Рисуем шестиугольники фиолетовым на третьей оси.
    for h in hexagons:
        ax3.add_patch(Polygon(h, closed=True, ec='purple', lw=2))
    # Применяем настройку осей.
    setup_axes(ax3)
    # Устанавливаем заголовок для подграфика.
    ax3.set_title('9.1. Шестиугольники (zip_polygons)')
    # Оптимизируем расположение подграфиков.
    plt.tight_layout()
    # Отображаем фигуру.
    plt.show()

    # Определяем генератор 2D точек с заданным шагом (пункт 9.2).
    def count_2D(start=(0, 0), step=(1, 1)):
        """Генератор 2D точек с заданным шагом"""
        # Получаем начальные координаты x, y.
        x, y = start
        # Бесконечный цикл для генерации точек.
        while True:
            # Возвращаем текущую точку (x, y).
            yield (x, y)
            # Увеличиваем x и y на соответствующие шаги.
            x += step[0]
            y += step[1]
    # Создаем генератор для верхних точек, начиная с (1,1) с шагом (3,0).
    gen_top    = count_2D(start=(1, 1), step=(3, 0))
    # Создаем генератор для нижних точек, начиная с (1,-1) с шагом (3,0).
    gen_bottom = count_2D(start=(1,-1), step=(3, 0))
    # Генерируем 6 верхних точек.
    top_vertices    = [next(gen_top)    for _ in range(6)]
    # Генерируем 6 нижних точек.
    bottom_vertices = [next(gen_bottom) for _ in range(6)]
    # Создаем фигуру и ось для пункта 9.2.
    fig, ax = plt.subplots(figsize=(6,4))
    # Рисуем верхние точки синим цветом.
    ax.scatter(*zip(*top_vertices),    color='blue')
    # Рисуем нижние точки синим цветом.
    ax.scatter(*zip(*bottom_vertices), color='blue')
    # Устанавливаем заголовок графика.
    ax.set_title("9.2. Вершины (count_2D)")
    # Применяем настройку осей.
    setup_axes(ax)
    # Отображаем фигуру.
    plt.show()

    # Разбиваем верхние точки на треугольники, группируя по три.
    tri_top    = zip_tuple(top_vertices[::3],    top_vertices[1::3],    top_vertices[2::3])
    # Разбиваем нижние точки на треугольники, группируя по три.
    tri_bottom = zip_tuple(bottom_vertices[::3], bottom_vertices[1::3], bottom_vertices[2::3])
    # Создаем список отрезков, соединяющих верхние и нижние точки.
    segments = list(zip(top_vertices, bottom_vertices))
    # Создаем фигуру и ось для пункта 9.3.
    fig, ax = plt.subplots(figsize=(6,4))
    # Рисуем отрезки между верхними и нижними точками зеленым цветом.
    for p1, p2 in segments:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', lw=2)
    # Устанавливаем заголовок графика.
    ax.set_title("9.3. Отрезок (zip_tuple)")
    # Применяем настройку осей.
    setup_axes(ax)
    # Отображаем фигуру.
    plt.show()