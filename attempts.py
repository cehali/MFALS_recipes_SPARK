import pickle

fname1 = 'ratings_train.txt'
with open(fname1, "rb") as fp1:
    ratings1 = pickle.load(fp1)

fname2 = 'ratings_test.txt'
with open(fname2, "rb") as fp2:
    ratings2 = pickle.load(fp2)

ratings = ratings1 + ratings2

file = open('ratings_full.txt', 'w')
pickle.dump(ratings, file)
file.close()
