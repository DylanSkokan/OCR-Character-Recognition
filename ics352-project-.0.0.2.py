import os
import platform
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from urllib.request import urlretrieve
from sklearn.preprocessing import LabelEncoder

# paths and directories
project_root = os.path.dirname(__file__)
data_directory = os.path.join(project_root, "batch_1")
images_directory = os.path.join(data_directory, "background_images")
json_file_path = os.path.join(data_directory, "JSON", "kaggle_data_1.json")

image_file_extenstions = ["jpg"]

# set directory to project root
os.chdir(project_root)

def clear_console ():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")      
# end clear_console
clear_console()

def get_dataframe_properties (frame):
    frame_length = len(frame)
    #print("type:", type(frame))
    null_value_count = frame.isnull().sum().sum()
    #dupes = frame.duplicated()
    #duplicate_row_count = print(type(frame.duplicated()))

    percent_null = null_value_count / frame_length * 100
    percent_duplicates = 0 #duplicate_row_count / frame_length * 100
    #print(frame.describe())
    #print(frame.head())
    
    print ("Column Information")
    print("Index Name\t\t\tData Type\t\tDimension")
    index = 0
    for column in frame.columns:
        print(index, " ", column, "\t\t\t\t", frame[column].dtype, "\t", frame[column].shape)
        index += 1
        
    print("percent nulls:", percent_null)
    print("percent duplicates:", percent_duplicates)
    print("total elements:", frame.size)
    print("row count", len(frame))
    print("width:", len(frame.columns))
    
    """
    print("\nSummary Statistics")
    print("-------------------")
    print(frame.describe())
    """
    print("First Five Rows")
    print("-------------------")
    print(frame.head())
    
    #print("Value Counts")
    #print("------------")
    #print(frame.value_counts())
# end get_dataframe_properties

def print_unique_values (frame):
    for column in frame.loc[:, frame.columns != "image_data"]: #columns:
        print(frame[column].value_counts)
# end print_unique_values

def print_column_descriptions ():
    print("Column Descriptions")
    print("-------------------")
    string = "latex: String--the ground truth latex"
    string += "\n" + "uuid: String"
    string += "\n" + "unicode_str: String--latex as unicode per latex_unicode_map.json"
    string += "\n" + "unicode_less_curlies: String--latex as unicode with curlies removed"
    string += "\n" + "font: String--identifier for the generating font"
    string += "\n" + "filename: String--filename of corresponding image"

    
    image_object_fields = "image_data: {"
    image_object_fields += "\n\t" + "full_latex_chars: List of Strings--each LaTeX token"
    """
        string += "\n" + "visible_latex_chars: List of Strings--only visible LaTeX tokens"
        "visible_char_map": List of Ints--LaTeX tokens per visible_char_map.json, 
        "width": Int--number of pixels of width of image, 
        "height": Int--number of pixels of height of image, 
        "depth": Int--channels of image RGBA, 
        "xmins": List of Floats--normalized position of xmin of bounding box per character, 
        "xmaxs": List of Floats, 
        "ymins": List of Floats, 
        "ymaxs": List of Floats, 
        "xmins_raw": List of Ints--pixel position of xmin of bounding box per character, 
        "xmaxs_raw": List of Ints,
        "ymins_raw": List of Ints,
        "ymaxs_raw": List of Ints,
        "png_masks": List of Strings--the encoded mask per character
        },
    """
    string += "\n" + image_object_fields
    print(string)
# end print_column_description
#print_column_descriptions()
def drop_extraneous_columns (frame, columns):
    for column in columns:
        print("Dropping", column)
        frame.drop(column, axis="columns", inplace=True)
# end drop_extraneous_columns

def convert_column_to_string ():
    #df[["latex", "uuid", "unicode_str"]] =  df[["latex", "uuid", "unicode_str"]].astype("string") 
    #df[["unicode_str", "unicode_less_curlies"]] =  df[["unicode_str", "unicode_less_curlies"]].astype("unicode") 
    df[["latex", "font", "filename"]] =  df[["latex", "font", "filename"]].astype("string")    
    for column in df.columns:
        print(column, "", df[column].dtype)
        print(df[column])
# end convert_column-to_string   

def label_features ():
    font_label_encoder = LabelEncoder()
    df["font_label"] = font_label_encoder.fit_transform(df["font"])
    print(df["font_label"].values)
#end label_features

# load json file into dataframe then get dataframe properties
df = pd.read_json(json_file_path)
get_dataframe_properties(df)
# drop uuid and unicode columns
drop_extraneous_columns(df, ["uuid", "unicode_str", "unicode_less_curlies"])
convert_column_to_string()

print_column_descriptions()
label_features()

# check what uuid column contains to verify it can be dropped

# 210 unique fonts.  The y might be clustered or they might need a k-clustering
print(df["font"].unique().shape)
print(df["latex"].unique().shape)

#print(df["unicode_str"], " ", df["unicode_less_curlies"])

# The most direct target is the latex expression.  We should be able to drop all other columns except latex and compare it to the picture metadata.
# The file name will help us test our model
print(df["font_label"])

"""
fig, ax = plt.subplots()
df["font_label"].value_counts().plot(ax=ax, kind="bar")
plt.show()
"""




# Extract the label column
font_labels = df.pop("font_label").values

# Normalize the feature columns using standardization
df = (df - df.mean()) / df.std()

# Determine the number of clusters
k = 20

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df)

# Evaluate the clusters using silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(df, clusters)

print(f"Silhouette score: {score}")








