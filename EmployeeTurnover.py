import pandas as pd
# import seaborn as sns
import seaborn as sns
from matplotlib import pyplot as plt

import tkinter as tk

from tkinter import filedialog
import webbrowser


def Csv_Viewer():

    root = tk.Tk() #create a object

    root.geometry("630x700+400+100")  # 630 is length and 700px is height
    root.title("Enter CsvFile")  # i can set the title with help of root.title
    root.configure(bg="blue")  # i can set the color middle portion of the csv viewer

    label = tk.Label(root, text="Click the button below to open a CSV file.",width=40, font="arial 20")
    label.pack(pady=10)

    click_open_button = tk.Button(root, text="Browse Csv", command=open_csv_file ,width=30, font="arial 10")
    click_open_button.pack(pady=5)

    root.mainloop() # in root.mainlop pdf viewer structure are ready


def open_csv_file():
    myfile_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if myfile_path:
        webbrowser.open(myfile_path)
        
    df = pd.read_csv(myfile_path)
    col_name = df.columns.tolist()
    print('Column name: ')
    print(col_name)    #we can print the coulumn

    print('\nSample data')
    df.head()

    tai = df.tail()
    print("last data of dataset: ",tai)
    
    s= df.shape
    print("we cane see the shape of dataset: ",s) #we can see shape of the dataset how many row and coulumn

    colm = df.columns
    print("\ntotal columns: ",colm)

    su= df.isna().sum() #It return sum of the missing value
    print("\n",su)

    dupli = df.duplicated(keep="first").sum() #we can check any any duplicate value prresnt in dataset or not
    print("\nduplica: ",dupli) 
    
    df.drop_duplicates(inplace=True) #we can remove the duplicate data by using  drop_uplicate method 
    sh = df.shape # we can print te data after removing duplicate data in dataset
    print("\n",sh)

    inf = df.info() # we can check the Datatype in dataset
    print("\ninformation: ",inf) 

   # Data Analysis explore krenege
    avg_mnthly = df["average_montly_hours"].unique() # we can check how any unique row
    print("\nAvg monthly hours: ",avg_mnthly) 

    bins = len(df["average_montly_hours"].unique())
    print("\n",bins)
    
    # Average Monthy Hours
    #We can plot the grapgh
    plt.figure(figsize=(6,6)) #we can fix the size 6 by 6 tuple 
    sns.histplot(data=df, x= df["average_montly_hours"], kde=True) #we can fill the graph data
    plt.tight_layout() #Graph look better
    plt.show() #We can see the gragh ny using show() method

   # Number of Projects
   #show the data number of project each employee done
    number_prjct = df["number_project"].value_counts()
    print("\n",number_prjct)

    plt.figure(figsize=(6,6)) #we can fix the size 6 by 6 tuple 
    sns.countplot(data=df, x="number_project")
    plt.title("Number of Projects Undertaken Rate") # We can set the title of the grapgh that show in grapgh
    plt.tight_layout() #used to automatically adjust subplot parameters to give specified padding
    plt.show() #We can see the gragh ny using show() method
    
    # Promotion made in last 5 year
    prom = df["promotion_last_5years"].value_counts() #It show how many emloyee are promoted in the last five year
    print("\n",prom)
    lables = df["promotion_last_5years"].value_counts().index.tolist() #in this we can see the lables 0,1
    print("\nLables: ",lables)

    lables = ["Promoted" if i==1 else "Not Promoted" for i in lables] # see the lables lable(0,1) 1 mean promoted and 0 mean not promoted
    print("\n",lables)

    plt.figure(figsize=(6,6))
    plt.pie(x= df["promotion_last_5years"].value_counts(),labels=lables, autopct="%1.2f%%", explode=[0,0.3]) #we can fill the graph data
    plt.title("Promotion in last the last five years") # We can set the title of the grapgh that show in grapgh
    plt.tight_layout() #Automatically adjust grapgh parameter and padding and graph look better 
    plt.show()#We can see the gragh ny using show() method

    # Work Accidents
    accdnt = df["Work_accident"].value_counts()#Yha values show 1 par aur 0 pr
    print("\n",accdnt)

    labels = df["Work_accident"].value_counts().index.tolist() #we can see the lables(0,1)
    print(labels)

    labels = ["Work Accident" if i==1 else "No Work Accident" for i in labels] # if value is 1 work accident else mean not work accident
    print("\n",labels)

    plt.figure(figsize=(6,6)) #we can fix the size 6 by 6 tuple 
    plt.pie(x= df["Work_accident"].value_counts(),labels=labels, autopct="%1.2f%%", explode=[0,0.3])
    plt.title("Work Accident") # We can set the title of the grapgh that show in grapgh
    plt.tight_layout() #Automatically adjust krea parameter and padding  graph look better 
    plt.show() #We can see the gragh ny using show() method
    
    # Time Spent in the company
    time = df["time_spend_company"].value_counts() 
    print("\n",time)#We can print the and count it
    labels = df["time_spend_company"].value_counts().index.tolist()#I this part we define a lables
    print("\n",labels)#We can print the lables

    plt.figure(figsize=(6,6))#we can fix the size 6 by 6 tuple 
    sns.countplot(x= "time_spend_company", data=df) 
    plt.title("Number of years Spent working in company")# We can set the title of the grapgh that show in grapgh
    plt.tight_layout() #Automatically adjust krea parameter and padding  graph look better
    plt.show() #We can see the gragh ny using show() method
    men = df["time_spend_company"].mean() # we can print mean avg year  emplyoyee stay in the company
    print("\nmean value ",men)
    
    # Department Comparison
    drpartment = df["sales"].unique()
    print("\n department",drpartment)#we can print all the department
    cunt = df["sales"].value_counts() # Yha values ko count kiya kis department me kitne employee hai #we can count the values
    print("\ncount",cunt) #we can count a value and print it

    plt.figure(figsize=(6,6)) #we can fix the size 6 by 6 tuple 
    sns.countplot(x="sales", data=df) # we can plot the grapgh
    plt.title("Number of Employee Per Department") # We can set the title of the grapgh that show in grapgh
    plt.xticks(rotation=90) #we can set the rotation of the graph
    plt.tight_layout() #Automatically adjust krea parameter and padding  graph look better
    plt.show()#We can see the gragh ny using show() method
    
    # Salary 
    slry = df["salary"].unique()
    print("\nSalary",slry)

    slry_count = df["salary"].value_counts() #we can print the salary data such as low salry,high salry,medium salry
    print("\nSalary Count",slry_count)

    labels = df["salary"].value_counts().index.tolist() #we can get the salary lables
    print("\nLables",labels)

    plt.figure(figsize=(6,6)) #we can fix the size 6 by 6 tuple 
    plt.pie(x= df["salary"].value_counts(),labels=labels, autopct="%1.2f%%") #Graph show hone ke liye data fill kiya
    plt.title("Salary Category")  # We can set the title of the grapgh that show in grapgh
    plt.tight_layout() #Automatically adjust krea parameter and padding  graph look better
    plt.show()#We can see the gragh ny using show() method
    
    # Data Preprocessing
    clumn= df.rename(columns={"sales": "department", "salary_leve": "salary_level"}, inplace=True) #Column rename kiya
    # print("\ncolum",clumn)

    myclm =df.columns 
    print("\n",myclm) #in this part i can print the column

   # Categogorical Encoding
    categogorical_cols =["department", "salary"]
    encoded_cols = pd.get_dummies(df[categogorical_cols], prefix="cat") #dummy data
    # print("\n",encoded_cols)
    df = df.join(encoded_cols) # join dummy column
    # print("\n",df)
    s = df.head()
    print("\n",s) #data print kara check kiya data add hua h ki nhi

    he= df.drop(["department","salary"],inplace=True,axis="columns") #we can  drop data inplcae =True mean data drop ho jaye
    # print(he)
    cul = df.columns # after drop we can printbthe column
    print("\n",cul)
    
    # Min Max Scaling
    #Min max scalling is used to improve our model and check how model are performed
    minimum =df["average_montly_hours"].min()
    print("\nMinimum hours",minimum) #Minumum hours

    maximum =df["average_montly_hours"].max() #Maximum hours
    print("\nMaxmum hours",maximum)

   #  Split Data Into Train and Test
    X = df.drop("left",axis=1)
    Y = df["left"]
    h = X.head()
    print("\n",h)

    yax = Y.head() #Yha data o aur 1 me show hoga 0 mean not left the company 1 mean left the company
    print("\n",yax)
    
    
    # Import Sklearn Model
    from sklearn.model_selection import train_test_split

    #Yha train aur test data liya
    # we can take data 80% for tranning and 20% of testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    mytrin = X_train.shape #Traning ke liye data ka shape dekha kitna number of data aa rha h traning ke liye
    print("\nX_Train ",mytrin)

    mytest = X_test.shape #Testing ke liye data ka shape dekha kitna number of data aa rha h testing ke liye
    print("\nX_test ",mytest)
    
    # Train Model
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    #Logistic regresion model
    logReg = LogisticRegression(max_iter=2000)#Maximum itration diya
    logReg.fit(X_train,Y_train)#we can fit the data
    logReg_prediction= logReg.predict(X_test)
    pred = accuracy_score(Y_test,logReg_prediction) #check the model accurecy
    print("\nPridition",pred)
    print(classification_report(Y_test,logReg_prediction)) #we can print precision, recall and all other classification report

    cm = confusion_matrix(Y_test, logReg_prediction)
    print("\n",cm)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    ranForest = RandomForestClassifier(n_estimators=100)
    ranForest.fit(X_train,Y_train) #we can fit the data
    ranForest_Prediction = ranForest.predict(X_test) #Yaha se predection kiya model kitna accurate h
    print("\nAccurecyScore ",accuracy_score(Y_test,ranForest_Prediction)) # we can check accurecy kiya
    print(classification_report(Y_test,ranForest_Prediction)) # we can print precision, recall and classifaction report

  

Csv_Viewer()#calling  main function to show output





