# PySpark Product-Category Relationship Solution

**Профессиональное решение задачи с продуктами и категориями на PySpark**

## 📋 Описание задачи

В PySpark приложении датафреймами (`pyspark.sql.DataFrame`) заданы продукты, категории и их связи. Каждому продукту может соответствовать несколько категорий или ни одной. А каждой категории может соответствовать несколько продуктов или ни одного. 

**Задача:** Написать метод на PySpark, который в одном датафрейме вернет:
- Все пары «Имя продукта – Имя категории»
- Имена всех продуктов, у которых нет категорий

## 🎯 Решение

### Основной подход

Используется **LEFT JOIN** для объединения таблиц, что гарантирует включение всех продуктов, даже тех, у которых нет категорий.

```python
def get_products_with_categories(self, products_df, categories_df, product_categories_df):
    # Шаг 1: LEFT JOIN продуктов со связями продукт-категория
    products_with_relations = products_df.alias("p") \
        .join(product_categories_df.alias("pc"), 
              col("p.product_id") == col("pc.product_id"), "left")
    
    # Шаг 2: LEFT JOIN с категориями для получения названий категорий
    result_df = products_with_relations.alias("pr") \
        .join(categories_df.alias("c"), 
              col("pr.category_id") == col("c.category_id"), "left") \
        .select(col("pr.product_name").alias("product_name"),
                col("c.category_name").alias("category_name"))
    
    return result_df
```

### Структура данных

#### Продукты (products_df)
| product_id | product_name |
|------------|--------------|
| 1 | iPhone 14 |
| 2 | MacBook Pro |
| ... | ... |

#### Категории (categories_df)
| category_id | category_name |
|-------------|---------------|
| 1 | Смартфоны |
| 2 | Ноутбуки |
| ... | ... |

#### Связи (product_categories_df)
| product_id | category_id |
|------------|-------------|
| 1 | 1 |
| 1 | 3 |
| ... | ... |

### Результат
| product_name | category_name |
|--------------|---------------|
| iPhone 14 | Смартфоны |
| iPhone 14 | Электроника |
| MacBook Pro | Ноутбуки |
| MacBook Pro | Электроника |
| Продукт без категории | NULL |

## 🚀 Запуск решения

### Предварительные требования

```bash
# Установка зависимостей
pip install -r requirements.txt
```

### Запуск основного решения

```bash
python pyspark_product_category_solution.py
```

### Запуск тестов

```bash
# Запуск всех тестов
pytest test_pyspark_solution.py -v

# Запуск конкретного теста
pytest test_pyspark_solution.py::TestProductCategoryProcessor::test_products_without_categories_included -v
```

### Системные требования

```bash
# Убедитесь что установлена Java 17+
export JAVA_HOME=/opt/homebrew/opt/openjdk@17  # macOS с Homebrew
export PATH="$JAVA_HOME/bin:$PATH"
java -version  # Должно показать Java 17+
```

## 📁 Структура проекта

```
.
├── pyspark_product_category_solution.py  # Основное решение
├── test_pyspark_solution.py             # Комплексные тесты
├── requirements.txt                      # Зависимости
└── README.md                            # Подробная документация
```

## 🧪 Тестирование

Проект включает комплексные тесты:

- ✅ **Базовая функциональность**: Проверка основного метода
- ✅ **Продукты без категорий**: Включение продуктов с NULL категориями
- ✅ **Множественные категории**: Обработка продуктов с несколькими категориями
- ✅ **Граничные случаи**: Работа с пустыми датафреймами
- ✅ **Пользовательские данные**: Тестирование на кастомных данных

## 💡 Ключевые особенности решения

### 1. Полнота результата
- **Все продукты включены**: Даже продукты без категорий
- **Все связи учтены**: Продукты с множественными категориями

### 2. Производительность
- **Оптимизированные JOIN'ы**: Использование LEFT JOIN для минимизации объема данных
- **Правильные алиасы**: Избежание конфликтов имен столбцов
- **Spark оптимизации**: Включена адаптивная оптимизация

### 3. Надежность
- **Обработка NULL значений**: Корректная работа с продуктами без категорий
- **Типизированные схемы**: Явное определение типов данных
- **Комплексное тестирование**: Покрытие всех сценариев

### 4. Читаемость кода
- **Подробные комментарии**: Объяснение каждого шага
- **Структурированный код**: Разделение на логические методы
- **Профессиональное оформление**: Следование PEP 8



## 📊 Альтернативные подходы

### SQL подход
```sql
SELECT 
    p.product_name,
    c.category_name
FROM products p
LEFT JOIN product_categories pc ON p.product_id = pc.product_id
LEFT JOIN categories c ON pc.category_id = c.category_id
```

### DataFrame API подход (основное решение)
```python
result_df = products_df.alias("p") \
    .join(product_categories_df.alias("pc"), 
          col("p.product_id") == col("pc.product_id"), "left") \
    .join(categories_df.alias("c"), 
          col("pc.category_id") == col("c.category_id"), "left") \
    .select(col("p.product_name"), col("c.category_name"))
```

### Использование SQL подхода (альтернатива)
```python
# Зарегистрировать DataFrame'ы как временные таблицы
products_df.createOrReplaceTempView("products")
categories_df.createOrReplaceTempView("categories")
product_categories_df.createOrReplaceTempView("product_categories")

# Выполнить SQL запрос
result_df = spark.sql("""
    SELECT p.product_name, c.category_name
    FROM products p
    LEFT JOIN product_categories pc ON p.product_id = pc.product_id
    LEFT JOIN categories c ON pc.category_id = c.category_id
""")
```

---

**Решение подготовлено профессиональным Python разработчиком**
**Технологии:** PySpark • Python • SQL • Apache Spark 