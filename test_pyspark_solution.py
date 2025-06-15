#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для PySpark решения задачи с продуктами и категориями
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col

from pyspark_product_category_solution import ProductCategoryProcessor


class TestProductCategoryProcessor:
    """
    Класс для тестирования ProductCategoryProcessor
    """
    
    @pytest.fixture(scope="class")
    def processor(self):
        """Фикстура для создания экземпляра процессора"""
        processor = ProductCategoryProcessor()
        yield processor
        processor.stop()
    
    def test_create_sample_data(self, processor):
        """Тест создания демонстрационных данных"""
        products_df, categories_df, product_categories_df = processor.create_sample_data()
        
        # Проверяем что DataFrame'ы не пустые
        assert products_df.count() > 0
        assert categories_df.count() > 0
        assert product_categories_df.count() > 0
        
        # Проверяем структуру схем
        assert "product_id" in products_df.columns
        assert "product_name" in products_df.columns
        assert "category_id" in categories_df.columns
        assert "category_name" in categories_df.columns
    
    def test_get_products_with_categories_basic(self, processor):
        """Базовый тест основного метода"""
        products_df, categories_df, product_categories_df = processor.create_sample_data()
        
        result_df = processor.get_products_with_categories(
            products_df, categories_df, product_categories_df
        )
        
        # Проверяем что результат не пустой
        assert result_df.count() > 0
        
        # Проверяем структуру результата
        assert "product_name" in result_df.columns
        assert "category_name" in result_df.columns
    
    def test_products_without_categories_included(self, processor):
        """Тест что продукты без категорий включены в результат"""
        products_df, categories_df, product_categories_df = processor.create_sample_data()
        
        result_df = processor.get_products_with_categories(
            products_df, categories_df, product_categories_df
        )
        
        # Должны быть продукты с NULL в category_name
        products_without_categories = result_df.filter(col("category_name").isNull())
        assert products_without_categories.count() > 0
        
        # Проверяем что это именно "Продукт без категории"
        product_names = [row.product_name for row in products_without_categories.collect()]
        assert "Продукт без категории" in product_names
        assert "Еще один продукт без категории" in product_names
    
    def test_products_with_multiple_categories(self, processor):
        """Тест что продукты с несколькими категориями обрабатываются корректно"""
        products_df, categories_df, product_categories_df = processor.create_sample_data()
        
        result_df = processor.get_products_with_categories(
            products_df, categories_df, product_categories_df
        )
        
        # iPhone 14 должен быть в двух категориях
        iphone_records = result_df.filter(col("product_name") == "iPhone 14").collect()
        assert len(iphone_records) == 2
        
        # Проверяем категории iPhone
        iphone_categories = [row.category_name for row in iphone_records]
        assert "Смартфоны" in iphone_categories
        assert "Электроника" in iphone_categories
    
    def test_edge_case_empty_dataframes(self, processor):
        """Тест граничного случая с пустыми DataFrame'ами"""
        # Создаем пустые DataFrame'ы с правильной схемой
        products_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True)
        ])
        
        categories_schema = StructType([
            StructField("category_id", IntegerType(), True),
            StructField("category_name", StringType(), True)
        ])
        
        product_categories_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("category_id", IntegerType(), True)
        ])
        
        empty_products = processor.spark.createDataFrame([], products_schema)
        empty_categories = processor.spark.createDataFrame([], categories_schema)
        empty_relations = processor.spark.createDataFrame([], product_categories_schema)
        
        result_df = processor.get_products_with_categories(
            empty_products, empty_categories, empty_relations
        )
        
        # Результат должен быть пустым но с правильной схемой
        assert result_df.count() == 0
        assert "product_name" in result_df.columns
        assert "category_name" in result_df.columns
    
    def test_custom_data_scenario(self, processor):
        """Тест с пользовательскими данными"""
        # Создаем кастомные данные для теста
        products_data = [
            (1, "Тестовый продукт 1"),
            (2, "Тестовый продукт 2"),
            (3, "Продукт без категории")
        ]
        
        categories_data = [
            (1, "Тестовая категория A"),
            (2, "Тестовая категория B")
        ]
        
        relations_data = [
            (1, 1),  # Продукт 1 -> Категория A
            (1, 2),  # Продукт 1 -> Категория B
            (2, 1),  # Продукт 2 -> Категория A
            # Продукт 3 без категорий
        ]
        
        # Создаем DataFrame'ы
        products_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True)
        ])
        
        categories_schema = StructType([
            StructField("category_id", IntegerType(), True),
            StructField("category_name", StringType(), True)
        ])
        
        product_categories_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("category_id", IntegerType(), True)
        ])
        
        test_products = processor.spark.createDataFrame(products_data, products_schema)
        test_categories = processor.spark.createDataFrame(categories_data, categories_schema)
        test_relations = processor.spark.createDataFrame(relations_data, product_categories_schema)
        
        result_df = processor.get_products_with_categories(
            test_products, test_categories, test_relations
        )
        
        # Ожидаем 4 записи: 
        # - Продукт 1 с категорией A и B (2 записи)
        # - Продукт 2 с категорией A (1 запись)
        # - Продукт 3 без категории (1 запись)
        assert result_df.count() == 4
        
        # Проверяем продукт без категории
        no_category = result_df.filter(
            (col("product_name") == "Продукт без категории") & 
            col("category_name").isNull()
        )
        assert no_category.count() == 1


def test_main_function():
    """Тест главной функции"""
    from pyspark_product_category_solution import main
    
    # Просто проверяем что функция выполняется без ошибок
    try:
        result = main()
        assert result is not None
    except Exception as e:
        pytest.fail(f"main() function failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 