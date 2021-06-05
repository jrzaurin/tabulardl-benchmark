mkdir raw_data

# Adult Census
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
mkdir raw_data/adult
mv adult.* raw_data/adult/

# Bank Marketting
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
mkdir raw_data/bank_marketing
mv bank-additional.zip raw_data/bank_marketing
cd raw_data/bank_marketing
unzip bank-additional.zip
mv bank-additional/* .
rm -r -f bank-additional/
cd ~/Projects/tabulardl-benchmark/

kaggle datasets download -d janiobachmann/bank-marketing-dataset
mkdir raw_data/bank_marketing_kaggle
mv bank-marketing-dataset.zip raw_data/bank_marketing_kaggle
cd raw_data/bank_marketing_kaggle
unzip bank-marketing-dataset.zip
cd ~/Projects/tabulardl-benchmark/

# Airbnb listings
wget http://data.insideairbnb.com/united-kingdom/england/london/2021-02-09/data/listings.csv.gz
mkdir raw_data/airbnb
mv listings.csv.gz raw_data/airbnb

# NYC Taxi Trip Duration
# I manually downloaded train_extended.csv from here:
# https://www.kaggle.com/neomatrix369/nyc-taxi-trip-duration-extended
kaggle competitions download -c nyc-taxi-trip-duration
mkdir raw_data/nyc_taxi
mv nyc-taxi-trip-duration.zip raw_data/nyc_taxi
cd raw_data/nyc_taxi
unzip nyc-taxi-trip-duration.zip
cd ~/Projects/tabulardl-benchmark/

# Forest Cover
wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
mkdir raw_data/covertype
mv covtype.data.gz raw_data/covertype
cd raw_data/covertype
gunzip covtype.data.gz
cd ~/Projects/tabulardl-benchmark/

# Facebook Volume
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip
mkdir raw_data/fb_comments
mv Dataset.zip raw_data/fb_comments
cd raw_data/fb_comments
unzip Dataset.zip
mv Dataset/Training/Features_Variant_5.csv .
cd ~/Projects/tabulardl-benchmark/
