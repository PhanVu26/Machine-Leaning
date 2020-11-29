from K_means.myKmeans import K_means

# so nhom can phan loai
K = 3

my_K_means = K_means(K)

file = "../datasets/k_means_data.csv"
#load du lieu( X : ma tran N rows)
X = my_K_means.load_data(my_K_means.K, file)

print("X", X)
#Hien thi du lieu
my_K_means.show_data(X)

# Tim center moi nhom
centers, y = my_K_means.kmeans(X, my_K_means.K)

#Hien thi ket qua
my_K_means.plot_result(X, y, centers, my_K_means.K, "Result")