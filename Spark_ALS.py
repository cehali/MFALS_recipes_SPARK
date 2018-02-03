import heapq
import pickle
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow
from pyspark.shell import spark, sqlContext, sc
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def generate_model_training():

    ratings = pd.read_csv('ratings.csv')

    ratings = spark.createDataFrame(ratings)

    string_indexer1 = StringIndexer(inputCol="user_id", outputCol="user_id_index")
    string_indexer2 = StringIndexer(inputCol="recipe_id", outputCol="recipe_id_index")

    indexers = [string_indexer1, string_indexer2]

    pipeline = Pipeline(stages=indexers)
    df_r = pipeline.fit(ratings).transform(ratings)

    (training, test) = df_r.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(rank=20, maxIter=20, regParam=0.1, userCol="user_id_index", itemCol="recipe_id_index", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))


def generate_model_complete():

    ratings = pd.read_csv('ratings.csv')

    ratings = spark.createDataFrame(ratings)

    string_indexer1 = StringIndexer(inputCol="user_id", outputCol="user_id_index")
    string_indexer2 = StringIndexer(inputCol="recipe_id", outputCol="recipe_id_index")

    indexers = [string_indexer1, string_indexer2]

    pipeline = Pipeline(stages=indexers)
    ratings_final = pipeline.fit(ratings).transform(ratings)

    # Build the recommendation model using ALS on the complete data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(rank=20, maxIter=20, regParam=0.1, userCol="user_id_index", itemCol="recipe_id_index", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(ratings_final)

    model.write().overwrite().save('recommendation_model_complete')

    # ratings_final.write.parquet('ratings_final')


def get_recommendation(user_index):

    ratings = pd.read_csv('ratings.csv')

    ratings_pivot = ratings.pivot_table(values='rating', index=['user_id'], columns=['recipe_id'], fill_value=0,
                        dropna=False)

    ratings_values = ratings_pivot.values

    recipes = np.load('recipes.npy')
    #ratings_values = np.load('ratings_values.npy')
    #ratings_pivot = pd.read_csv('ratings_pivot.csv')

    model = ALSModel.load('recommendation_model_complete')

    ratings = sqlContext.read.parquet('ratings_final')

    users_recs = model.recommendForAllUsers(10)

    recipes_recommended_list = users_recs.where(users_recs.user_id_index == user_index).select('recommendations')

    recipes_recommended = [i.recommendations for i in recipes_recommended_list.collect()]

    recommendations = []
    for rec in recipes_recommended[0]:
        result = ratings.where(ratings.recipe_id_index == rec.recipe_id_index).select('recipe_id')
        recommendations.append(result.rdd.flatMap(list).first())

    recipes_rated_index = heapq.nlargest(20, range(len(ratings_values[user_index])), ratings_values[user_index].take)
    # recipes_rated_index = [x + 1 for x in recipes_rated_index]
    recipes_rated_id = ratings_pivot.columns[[recipes_rated_index]]

    recipes_rated = []
    recipes_recom = []
    for rec1, rec2 in zip(recipes_rated_id, recommendations):
        for recipe in recipes:
            if rec1[1] == recipe.get('amazon_id'):
                recipes_rated.append(recipe)
            if rec2 == recipe.get('amazon_id'):
                recipes_recom.append(recipe)

    print ('Recipes you liked:')
    for i in range(0, len(recipes_rated)):
        print recipes_rated[i].get('title')

    print ('\nRecipes we recommend:')
    for i in range(0, len(recipes_recom)):
        print recipes_recom[i].get('title')


'''def get_similar_recipes_spark(recipe_index):

    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')

    recipes = pd.read_csv('/home/cezary/Dataset-recpies+ratings/epi_r_Epicurious.csv')

    recipes = recipes.drop('rating', 1)
    recipes = recipes.drop('calories', 1)
    recipes = recipes.drop('protein', 1)
    recipes = recipes.drop('fat', 1)
    recipes = recipes.drop('sodium', 1)
    recipes = recipes.drop('title', 1)
    # recipes.set_index('title', inplace=True)
    print recipes

    recipes1 = recipes.T

    print recipes1

    df = spark.createDataFrame(recipes1)

    rdd = df.rdd

    rdd = rdd.map(lambda data: Vectors.dense(data))

    mat = RowMatrix(rdd)

    # mat = IndexedRowMatrix(rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))

    # mat = mat.toRowMatrix()

    sims = mat.columnSimilarities()

    sims2 = sims.entries

    cord1 = []
    cord2 = []
    values = []

    for entr in sims2.collect():
        cord1.append(int(entr.i))
        cord2.append(int(entr.j))
        values.append(float(entr.value))


    similarities_list = pd.DataFrame(
        {'index1': cord1,
         'index2': cord2,
         'similarity': values
         })

    similar_recipes = similarities_list[similarities_list['index1'] == recipe_index]

    most_similar = similar_recipes.nlargest(10, 'similarity')

    print most_similar
    #print recipes[recipes.index == recipe_index].tolist()
    #print recipes[recipes.index == 650].tolist()'''
#generate_model_training()
generate_model_complete()
get_recommendation(1)
# get_similar_recipes_spark(2)
