from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


class ProductCategoryProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("ProductCategoryProcessor") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
    
    def create_sample_data(self):        
        # Схема для продуктов
        products_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True)
        ])
        
        # Схема для категорий
        categories_schema = StructType([
            StructField("category_id", IntegerType(), True),
            StructField("category_name", StringType(), True)
        ])
        
        # Схема для связей продукт-категория
        product_categories_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("category_id", IntegerType(), True)
        ])
        
        # Данные продуктов
        products_data = [
            (1, "iPhone 14"),
            (2, "MacBook Pro"),
            (3, "Samsung Galaxy S23"),
            (4, "Dell XPS 13"),
            (5, "Беспроводные наушники"),
            (6, "Продукт без категории"),
            (7, "Еще один продукт без категории")
        ]
        
        # Данные категорий
        categories_data = [
            (1, "Смартфоны"),
            (2, "Ноутбуки"),
            (3, "Электроника"),
            (4, "Аксессуары"),
            (5, "Неиспользуемая категория")
        ]
        
        # Связи продукт-категория (продукты 6 и 7 без категорий)
        product_categories_data = [
            (1, 1), (1, 3), (2, 2), (2, 3), (3, 1), 
            (3, 3), (4, 2), (5, 4), (5, 3)
        ]
        
        # Создание DataFrame'ов
        products_df = self.spark.createDataFrame(products_data, products_schema)
        categories_df = self.spark.createDataFrame(categories_data, categories_schema)
        product_categories_df = self.spark.createDataFrame(product_categories_data, product_categories_schema)
        
        return products_df, categories_df, product_categories_df
    
    def get_products_with_categories(self, products_df, categories_df, product_categories_df):
        # Шаг 1: LEFT JOIN продуктов со связями продукт-категория
        # Это гарантирует, что все продукты будут включены, даже без категорий
        products_with_relations = products_df.alias("p") \
            .join(
                product_categories_df.alias("pc"), 
                col("p.product_id") == col("pc.product_id"), 
                "left"
            )
        
        # Шаг 2: LEFT JOIN с категориями для получения названий категорий
        # Используем LEFT JOIN чтобы сохранить продукты без категорий
        result_df = products_with_relations.alias("pr") \
            .join(
                categories_df.alias("c"), 
                col("pr.category_id") == col("c.category_id"), 
                "left"
            ) \
            .select(
                col("pr.product_name").alias("product_name"),
                col("c.category_name").alias("category_name")
            )
        
        return result_df
    
    def display_results(self, result_df):
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТ: Все пары + продукты без категорий")
        print("=" * 50)
        result_df.show(truncate=False)
        print(f"Всего записей: {result_df.count()}")
    
    def run_demonstration(self):
        print("Создание демонстрационных данных...")
        products_df, categories_df, product_categories_df = self.create_sample_data()
        
        print("\nИсходные данные:")
        print("\n1. ПРОДУКТЫ:")
        products_df.show()
        
        print("2. КАТЕГОРИИ:")
        categories_df.show()
        
        print("3. СВЯЗИ ПРОДУКТ-КАТЕГОРИЯ:")
        product_categories_df.show()
        
        print("\nВыполнение основного метода...")
        result_df = self.get_products_with_categories(products_df, categories_df, product_categories_df)
        
        self.display_results(result_df)
        
        return result_df
    
    def stop(self):
        self.spark.stop()


def main():
    processor = ProductCategoryProcessor()
    
    try:
        # Запускаем демонстрацию
        result = processor.run_demonstration()
        
        print("\n" + "="*40)
        print("✅ РЕШЕНИЕ ВЫПОЛНЕНО!")
        print("="*40)
        
        return result
        
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
        raise
    finally:
        processor.stop()


if __name__ == "__main__":
    main() 