Реализовать api, которое позволяет: генерировать, преобразовывать и визуализировать последовательность плоских полигонов, представленных в виде кортежа кортежей (например: ((0,0), (0,1), (1,1), (1,0)) – представление для квадрата). Последовательности представлений полигонов представляют из себя итераторы (далее: последовательности полигонов). Решать задачи с использованием функционального стиля программирования, в том числе активно использовать функции из модуля itertools и functools.
### 1.	Реализовать функцию визуализации последовательности полигонов, представленной в виде итератора (например, можно использовать визуализацию с помощью библиотеки matplotlib, см. пример): https://matplotlib.org/stable/gallery/shapes_and_collections/patch_collection.html#sphx-glr-gallery-shapes-and-collections-patch-collection-py 

### 2.	Реализовать функции, генерирующие бесконечную последовательность не пересекающихся полигонов с различающимися координатами (например, «ленту»):

    1.	прямоугольников ( gen_rectangle() );
    2.	треугольников ( gen_triangle() );
    3.	правильных шестиугольников ( gen_hexagon() ).
    4.	с помощью данных функций используя функции из модуля itertools сгенерировать 7 фигур, включающих как прямоугольники, так и треугольники и шестиугольники, визуализировать результат.
<img width="560" alt="image" src="https://github.com/user-attachments/assets/65098c43-99cb-4a1f-ba28-f1431aeba1ea" />

### 3.	Реализовать операции:
   ```
    1.	параллельный перенос ( tr_translate );
    3.	поворот ( tr_rotate );
    5.	симметрия ( tr_symmetry );   
    6.	гомотетия ( tr_homothety );
```
       
которые можно применить к последовательности полигонов с помощью функции map.
<img width="554" alt="image" src="https://github.com/user-attachments/assets/6f4fa532-9485-4824-9681-fd5859764a8f" />

### 4.	С помощью данных функций создать и визуализировать:
   ```
    1.	3 параллельных «ленты» из последовательностей полигонов, расположенных под острым углом к оси x;
    2.	две пересекающихся «ленты» из последовательностей полигонов, пересекающихся не в начале координат;
    3.	две параллельных ленты треугольников, ориентированных симметрично друг к другу;
    4.	последовательность четырехугольников в разном масштабе, ограниченных двумя прямыми, пересекающимися в начале координат (см. рис.).
  ```

### 5.	Реализовать операции: 

    1.	фильтрации фигур, являющихся выпуклыми многоугольниками ( flt_convex_polygon );
    2.	фильтрации фигур, имеющих хотя бы один угол, совпадающий с заданной точкой ( flt_angle_point );
    3.	фильтрации фигур, имеющих площадь менее заданной ( flt_square );
    4.	фильтрации фигур, имеющих кратчайшую сторону менее заданного значения ( flt_short_side );
    5.	фильтрации выпуклых многоугольников, включающих заданную точку (внутри многоугольника) ( flt_point_inside );
    6.	фильтрации выпуклых многоугольников, включающих любой из углов заданного многоугольника ( flt_polygon_angles_inside );
  
которые можно применить к последовательности полигонов с помощью функции filter.

### 6.	С помощью данных функций реализовать и визуализировать:
    1.	фильтрацию фигур, созданных в рамках пункта 4.4; подобрать параметры так, чтобы на выходе было получено 6 фигур;
    2.	используя функции генерации из п. 2 и операции из п. 3 создать не менее 15 фигур, которые имеют различный масштаб и выбрать из них (подбором параметра фильтрации) не более 4х фигур, имеющих кратчайшую сторону менее заданного значения;
    3.	используя функции генерации из п. 2 и операции из п. 3 создать не менее 15 фигур имеющих множество пересечений и обеспечить фильтрацию пересекающихся фигур.


### 7.	Реализовать декораторы и продемонстрировать корректность их работы:
    1.	Фильтрующие многоугольники в итераторах среди аргументов функции, работающие на основе функций из 5: @flt_convex_polygon, @flt_angle_point, @flt_square, @flt_short_side, @flt_point_inside, @flt_polygon_angles_inside ;
    2.	Преобразующие многоугольники в итераторах среди аргументов функции, работающие на основе функций из 3: @tr_translate, @tr_rotate, @tr_symmetry, @tr_homothety ;


### 8.	Реализовать функции и продемонстрировать их корректность: 
    1.	поиск угла, самого близкого к началу координат ( agr_origin_nearest );
    2.	поиск самого длинной стороны многоугольника ( agr_max_side ); 
    3.	поиск самой маленькой площади многоугольника ( agr_min_area );
    4.	расчет суммарного периметра ( agr_perimeter );
    5.	расчет суммарной площади ( agr_area ).
которые можно применить к последовательности полигонов с помощью функции functools.reduce .


### 9.	Реализовать функции и продемонстрировать пример их работы (если возможно, с визуализацией):
    1.	склейки полигонов в одну последовательность полигонов из нескольких последовательностей полигонов zip_polygons(iterator1, iterator2, [iterator3, …]). Пример:  zip_polygons([((1,1), (2,2), (3,1)), ((11,11), (12,12), (13,11))], [((1,-1), (2,-2), (3,-1)), ((11,-11), (12,-12), (13,-11))]) ->[((1,1), (2,2), (3,1), (1,-1), (2,-2), (3,-1)), ((11,11), (12,12), (13,11), (11,-11), (12,-12), (13,-11))] .
Альтернативный пример (визуализация):
<img width="542" alt="image" src="https://github.com/user-attachments/assets/132de030-2a6d-42df-b817-668a05e6f89e" />

    2.	генерации count_2D() параметры: (start1, start2), [(step1, step2)], результаты: (start1, start2), (start1+step1, start2+step2), (start1+2*step1, start2+2*step2)
    3.	склейки полигонов в одну последовательность полигонов из нескольких последовательностей zip_tuple(iterator1, iterator2) . Пример: zip_tuple([(1,1),  (2,2), (3,3), (4,4)], [(2,2), (3,3), (4,4), (5,5)], [(3,3), (4,4), (5,5), (6,6)]) -> ((1,1),  (2,2), (3,3)), ((2,2), (3,3) (4,4)), ((3,3), (4,4), (5,5)), ((5,5), (6,6), (7,7))



