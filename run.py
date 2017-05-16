from Sentiment import Sentiment

sentiment = Sentiment()

model = sentiment.train_model(load_data=True,nb_epoch=1,old_weight_path="output/weights.13-0.20.hdf5")


tests = ['发货太慢了，商家服务太差.','脑白金强势登陆央视']

labels = sentiment.predict_label(model,tests)

print(labels)
