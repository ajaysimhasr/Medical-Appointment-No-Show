
# coding: utf-8

# <img src="C:\\Users\\byabh\\Desktop\\pm\\project\\rbs.jpg" width="300">
# # Project: Medical Appointment No-Show
# ## Course: Python Methodologies for Data Science (PMDS) 
# ## Spring 2018, Rutgers Business School
# ## Professor: Lars Sorensen 
# ## Team: Abhilash Basuru Yethesh Kumar, Ajay Simha Subraveti Ranganatha, Annapoorna Chandrashekar Kadur, Supriya Nanjundaswamy
# 

# <img src="C:\\Users\\byabh\\Desktop\\pm\\project\\noshow.jpg" width="500">
# <img src="C:\\Users\\byabh\\Desktop\\pm\\project\\noshow1.jpg" width="650">

# #### List of required Python Machine learning Packages, Third party libraries for Statistical analysis and visualization

# In[1]:


import pandas as pd
import numpy as np
# Visualization
import seaborn as sns
# Plotting graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  


# #### Read data in csv file using read csv from pandas library

# In[2]:


data = pd.read_csv("C:/Users/byabh/Desktop/pm/project/noshow_appointments.csv")


# ### 1. Dataset Check

# #### 1.1 Check how many rows and columns we have in dataset

# In[3]:


data.shape


# #### We have a total of 110527 observations with 14 features in our dataset

# #### 1.2 Check information of each attribute

# In[4]:


data.info()


# #### 1.3 Check for any missing values

# In[5]:


data.isnull().sum()


# #### No missing values observed

# #### 1.4 Display the first five rows of the data

# In[6]:


data.head()


# ### 2. Data Cleaning

# #### 2.1. Correct the typos in column names - from the above table, we can see Hypertension as Hipertension and Handicap as Handcap. Renamed N0-show to NoShow

# In[7]:


data.rename(columns = {'Handcap': 'Handicap','Hipertension': 'Hypertension', 'No-show': 'NoShow'}, inplace = True)

print(data.columns)


# #### 2.2. It is always advisable  to keep uniform datatime format when working with date and time columns - convert the ScheduledDay and AppointmentDay columns into datetime64 format 

# In[8]:


data.ScheduledDay = data.ScheduledDay.apply(np.datetime64)
data.AppointmentDay = data.AppointmentDay.apply(np.datetime64)


# #### 2.3 Extract new features from the exisitng features - define two functions timecal and datesep.
# whichHour, ScheduledDayDate are the new features extracted by applying the functions on ScheduledDay and AppointmentDay columns.
# timecal will split the schedule day into hour, min, seconds and round the value to return in what hour of the day appoinment was schedulded,
# datesep function returns the day of the schedulded and appointment day

# In[9]:


def timecal(timestamp):
    timestamp = str(timestamp)
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    second = int(timestamp[17:])
    return round(hour + minute/60 + second/3600)

def datesep(day):
    day=str(day)
    day=str(day[:10])
    return day
data['whichHour'] = data.ScheduledDay.apply(timecal)
data['ScheduledDayDate'] = data.ScheduledDay.apply(datesep)
data['AppointmentDay'] = data.AppointmentDay.apply(datesep)


# #### 2.4 Calculate what day of the week is the appoinment and what is the difference in number of days between scheduled day and appointment day

# In[10]:


data['ScheduledDayDate']=data['ScheduledDayDate'].apply(np.datetime64)
data['ScheduledDayDate'] = pd.to_datetime(data['ScheduledDayDate'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['appointment_day'] = data['AppointmentDay'].dt.weekday_name


# In[11]:


appoint_day = pd.to_datetime(data.AppointmentDay)
schedul_day =  pd.to_datetime(data.ScheduledDay)
wait_time = appoint_day -schedul_day
data['days_difference'] = pd.DataFrame(wait_time)
data['days_difference'] =(data.days_difference/np.timedelta64(1, 'D')).astype(int)


# #### 2.5 Check for any erroneous values in the dataset

# In[12]:


print('Age:',sorted(data.Age.unique()))
print('Gender:',data.Gender.unique())
print('Neighbourhood:',data.Neighbourhood.unique())
print('Scholarship:',data.Scholarship.unique())
print('Hypertension:',data.Hypertension.unique())
print('Diabetes:',data.Diabetes.unique())
print('Alcoholism:',data.Alcoholism.unique())
print('Handicap:',data.Handicap.unique())
print('SMS_received:',data.SMS_received.unique())
print('whichHour:',data.whichHour.unique())
print('appointment_day:',data.appointment_day.unique())
print('NoShow:',data.NoShow.unique())


# #### No errored values observed in any columns expect for age. (age = -1, 100, 102, 115) Although we have instances of humans living for 100 or more years, we are treating these values as outliers in our analysis

# #### 2.5.1 Remove the outliers from the data

# In[13]:


data = data[(data.Age >= 0) & (data.Age <= 99)]


# #### 2.6 Look into the data after cleaning is completed.

# In[77]:


data.head()


# ### 3. Exploratory Data Analysis

# #### 3.1 Lets look into the number of show or no show cases in our data and plot them

# In[15]:


count = data[['AppointmentID','NoShow']].groupby('NoShow').count()
print(count)
plot1  = data[['AppointmentID','NoShow']].groupby('NoShow').count().plot(kind='bar',legend=False)


# #### 3.2 Build correlation matrix to see how each variables are correlated with each other

# In[16]:


corr = data[data.columns].corr()
corr


# In[17]:


f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax,annot=True)


# #### From correlation matrix, we can see that age is positively correlated with hypertension(0.5) meaning as age increases hypertension tends to increase by 50% and similarly diabetes has positive correlation of 0.29 with age.

# #### 3.3 Gender analysis - number of men and women who showed up or missed their appointments

# In[18]:


gender=data.groupby(['Gender','NoShow'])['NoShow'].size()
print(gender)
plot2= data.groupby('Gender')['NoShow'].value_counts(normalize = True).plot(kind='bar')


# #### We can see that women visit hospitals slightly more than men. But we don't have any data to analyize why this pattern.

# #### 3.4 Plot the number of men and women suffering from each of the medical problems in the data

# In[19]:


problem=data[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum()
print(problem)
plot3=data[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum().plot(kind='barh',figsize=(21,12))


# #### 3.5 create a new class feature_analysis which will:
# 1. The class feature analysis, takes the feature(Diabetes, Hypertension, etc) as input to the class followed by a constructor  and two methds: Visual and calc. 
# 2. visual class: Plots the countplot using the feature assigned from the dataframe 
# 3. calc class: Calculates what percentage of patients missed appointments

# In[20]:


class feature_analysis(object):
    def __init__(self,feature):
        self.feature = feature
    
    def visual(self,df):
        sns.countplot(self.feature,data=df,hue='NoShow',palette='viridis')

    def calc(self,df,x):
        percentage = (sum((df[self.feature]==x) & (df['NoShow']=='Yes'))/sum(df[self.feature]==x))*100
        print('The Percentage of {} patients not attending appointments is: {}%'.format(self.feature,round(percentage,2)))


# #### 3.6 Using the class written above, plot the graphs for percentage of each diseased pateint's show or no show

# In[21]:


f1 = feature_analysis('Diabetes')
f1.visual(data)
f1.calc(data,1)


# In[22]:


f2 = feature_analysis('Alcoholism')
f2.visual(data)
f2.calc(data,1)


# In[23]:


f3 = feature_analysis('Handicap')
f3.visual(data)
f3.calc(data,1)


# In[24]:


f4 = feature_analysis('Hypertension')
f4.visual(data)
f4.calc(data,1)


# #### 3.7 Which day of the week were more appointments booked?

# In[25]:


weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    j=data[data.appointment_day==i]
    count=len(j)
    total_count=len(data)
    perc=(count/total_count)*100
    print(i,count)
    plt.bar(index,perc)
plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.title('Day of the week for appointment')
plt.show()


# #### 3.8 Which day of the week were more appointments missed?

# In[26]:


no_Show_Yes=data[data['NoShow']=='Yes']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    k=no_Show_Yes[no_Show_Yes.appointment_day==i]
    count=len(k)
    total_count=len(no_Show_Yes)
    perc=(count/total_count)*100
    print(i,count,perc)
    plt.bar(index,perc)

plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.title('Percent of No-Show per DayOfWeek')
plt.show()


# #### 3.9 Lets look at the location of hospitals, which neighbourhood has more appointments?

# In[27]:


location=data.groupby(['Neighbourhood'],sort=False).size()
print(location.sort_values())
location_plot=data.groupby(['Neighbourhood']).size().plot(kind='bar',figsize=(20,10))
plt.xticks(rotation=90)


# #### 3.10 What percentage of men and women missed their appointments?

# In[28]:


Men_perc=data[(data['Gender']=='M') & (data['NoShow']=='Yes')].count()
Women_perc=data[(data['Gender']=='F') & (data['NoShow']=='Yes')].count()
Men_total=len((data['Gender']=='M'))
Women_total=len((data['Gender']=='F'))
percentage_of_Women=(Women_perc/Women_total)*100
percentage_of_men=(Men_perc/Men_total)*100
print("Percentage of women who missed their appointment: ",np.round(percentage_of_Women['Gender'],0),"%")
print("Percentage of men who missed their appointment:  ",np.round(percentage_of_men['Gender'],0),"%")


# #### 3.10.1 Lets visualize the total percentage of appointments missed in terms of 100% split between men and women

# In[29]:


labels='Female','Male'
sizes=[13,7]
plot = plt.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)


# #### 3.11 Scholarship analysis to no show

# In[30]:


sns.countplot(x='Scholarship',data=data,hue='Gender')
scholarship=data.groupby(['NoShow','Scholarship'])['Scholarship'].count()
print(scholarship)


# #### 3.12 NoShow analysis of the appointment_day and difference in appointment day to scheduled day

# In[31]:


sns.barplot(x ='appointment_day',y='days_difference',hue='NoShow', data=data)
plt.show()


# #### 3.13 Plot the age vs difference in appointment day to schedulded day

# In[32]:


g = sns.FacetGrid(data , hue='NoShow',size=7)
g.map(plt.scatter,'Age','days_difference', alpha = .7)
g.add_legend();
plt.show()


# #### 3.14 Define a function which will classify the age to Child, Adult and Senior

# In[33]:


def FormatAge (age):
    if age['Age']>0 and age['Age']<=17 :
        return 'Child'
    elif age['Age']>=18 and age['Age'] <50:
        return 'Adult'
    else:
        return 'Senior'


# #### 3.14.1  Classification of age based on the above classes.

# In[34]:


data['AgeClass'] = data.apply(FormatAge,axis=1)


# #### 3.15 Plot the age distribution in there respective age class and the show /no show of age class

# In[35]:


sns.set_style('darkgrid')

sns.countplot(data['AgeClass'], alpha =.80,palette="muted")
plt.title('Age Classes ')
plt.show()

print (data.groupby('Age')['NoShow'].value_counts(normalize = True))

sns.set_style('darkgrid')
fig = sns.countplot(x='AgeClass', data=data,hue='NoShow', palette="muted");
plt.show()


# #### 3.16 Plot of Show/Noshow to SMS Received

# In[36]:


ax = sns.countplot(x=data.SMS_received, hue=data.NoShow, data=data)
ax.set_title("Show/NoShow for SMSReceived")
x_ticks_labels=['No SMSReceived', 'SMSReceived']
ax.set_xticklabels(x_ticks_labels)
plt.show()


# #### 3.17 Scholorship Analysis for Show/NoShow

# In[37]:


df_s_ratio = data[data.NoShow == 'No'].groupby(['Scholarship']).size()/data.groupby(['Scholarship']).size()
ax = sns.barplot(x=df_s_ratio.index, y=df_s_ratio, palette="RdBu_r")
ax.set_title("Percentage for Scholarship")
x_ticks_labels=['No Scholarship', 'Scholarship']
ax.set_xticklabels(x_ticks_labels)
plt.show()


# From the above graph, we can see that 80% have come for the visit with no Scholarship and 75% came to visit with Scholarship.

# ### 4.Machine Learning Models

# #### 4.1 Create model with NoShow as the predictor variable. Convert categorical values to 0 and 1

# In[38]:


Y = data['NoShow']
Y = Y.map({'No': 0, 'Yes': 1})

X = data.drop(labels = ['NoShow', 'PatientId', 'AppointmentID'], axis = 1)
X['Neighbourhood'] = X['Neighbourhood'].astype('category').cat.codes
X['appointment_day'] = X['appointment_day'].astype('category').cat.codes
X['Gender'] = X['Gender'].map({'M': 0, 'F': 1})


# #### 4.1.1 Drop the columns which are not required for model analysis

# In[39]:


X = X.drop(labels = ['ScheduledDay', 'AppointmentDay', 'ScheduledDayDate','AgeClass'], axis = 1)


# #### 4.1.2 split the data into training and testing dataset in the ratio 75:25

# In[40]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# #### 4.2 LogisticRegression
# 
# Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).
# 
# In logistic regression, the dependent variable is binary or dichotomous, i.e. it only contains data coded as 1 (TRUE, success  etc.) or 0 (FALSE, failure etc.). In this case, the outcome is Show or No-Show.
# 
# The goal of logistic regression is to find the best fitting model to describe the relationship between the dichotomous characteristic of interest and a set of independent variables. Logistic regression generates the coefficients of a formula to predict a logit transformation of the probability of presence of the characteristic of interest
# 

# In[41]:


model1 = LogisticRegression()

model1.fit(x_train, y_train)


# In[42]:


predictions = model1.predict(x_test)


# In[43]:


score = model1.score(x_test, y_test)
print(score)


# In[44]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[45]:


print(classification_report(y_test, predictions))


# In[46]:


plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'gist_rainbow_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[47]:


df1=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})  
df1


# #### Logistic regression gives accuracy of 79.61% for this data. So, if a new patient data is given as input to this model, we can predict with 79% accuracy if that patient shows up or not for the appointment scheduled.

# #### FIle output

# In[48]:


df1.to_csv('LogisticRegression_classification.csv')


# #### 4.3 Random Forest
# 
# Random forest or random decision forest is an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. For our analysis random forest serves as a classifier for classifying if a patient shows-up.

# In[49]:


model2 = RandomForestClassifier(n_estimators = 10, max_depth = 10)

model2.fit(x_train, y_train)


# In[50]:


predictions1 = model2.predict(x_test)


# In[51]:


score1 = model2.score(x_test, y_test)
print(score1)


# In[52]:


cm1 = metrics.confusion_matrix(y_test, predictions1)
print(cm1)


# In[53]:


print(classification_report(y_test, predictions1))


# In[54]:


plt.figure(figsize=(6,6))
sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'gist_rainbow_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score1)
plt.title(all_sample_title, size = 15);


# In[55]:


df2=pd.DataFrame({'Actual':y_test, 'Predicted':predictions1})  
df2


# #### Random forest gives an accuracy of 79.97% which is better compared to Logistic Regression.

# #### 4.3 Naive Bayes Classification
# 
# In machine learning, Naive Bayes classifiers (sometimes called the idiot Bayes model) are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
# When dealing with continuous data, a typical assumption is that the continuous values associated with each class are distributed according to a Gaussian distribution. So, we are applying gausian naive bayes model to clasify show or noshow of patients.
# 

# In[56]:


model3 = GaussianNB()


# In[57]:


model3.fit(x_train, y_train)


# In[58]:


predictions2 = model3.predict(x_test)


# In[59]:


score2 = model3.score(x_test, y_test)
print(score2)


# In[60]:


cm2 = metrics.confusion_matrix(y_test, predictions2)
print(cm2)


# In[61]:


print(classification_report(y_test, predictions2))


# In[62]:


plt.figure(figsize=(6,6))
sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'gist_rainbow_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score2)
plt.title(all_sample_title, size = 15);


# In[63]:


df3=pd.DataFrame({'Actual':y_test, 'Predicted':predictions2})  
df3


# #### Naive bayes gives accuracy of 76.94% to predicit the output.

# #### 4.4 Decision Tree
# 
# Decision tree is largely used non-parametric effective machine learning modeling technique for regression and classification problems. To find solutions, decision tree makes sequential, hierarchical decision about the outcome variable based on the predictor data. Hierarchical means the model is defined by a series of questions that lead to a class label or a value when applied to any observation. Once set up, the model acts like a protocol in a series of “if this occurs then this occurs” conditions that produce a specific result from the input data.
# A Non-parametric method means that there are no underlying assumptions about the distribution of the errors or the data. It basically means that the model is constructed based on the observed data.

# In[64]:


model4 = tree.DecisionTreeClassifier()
model4.fit(x_train, y_train)


# In[65]:


predictions3 = model4.predict(x_test)


# In[66]:


score3 = model4.score(x_test, y_test)
print(score3)


# In[67]:


cm3 = (confusion_matrix(y_test, predictions3))  
print(cm3)


# In[68]:


print(classification_report(y_test, predictions3))


# In[69]:


plt.figure(figsize=(6,6))
sns.heatmap(cm3, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'gist_rainbow_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score3)
plt.title(all_sample_title, size = 15);


# In[70]:


df4=pd.DataFrame({'Actual':y_test, 'Predicted':predictions3})  
df4


# #### Decision tree predicts with accuracy of 71.89% .

# #### 4.5 K Means Clustering
# K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
# 
# The centroids of the K clusters, which can be used to label new data.
# Labels for the training data (each data point is assigned to a single cluster)
# Rather than defining groups before looking at the data, clustering allows you to find and analyze the groups that have formed originally.

# In[71]:


model5 = KMeans(n_clusters=2)
model5.fit(x_train, y_train)


# In[72]:


predictions4 = model5.predict(x_test)


# In[73]:


labels = KMeans(2, random_state=0).fit_predict(x_test)


# In[74]:


print(classification_report(y_test, predictions4))


# In[75]:


plt.scatter(x_test.iloc[:, 0].values, x_test.iloc[:, 1].values, c=labels,
            s=50, cmap='viridis');


# In[76]:


df5=pd.DataFrame({'Actual':y_test, 'Predicted':predictions4})  
df5


# ### 5. Conclusion

# 1. 79.79% did show up for the appointment whereas 20.2% of them did not.
# 2. Women visit hospitals more than men.
# 3. Hypertension is seen more in women which might be one of the reasons why women visit hospitals more than men.
# 4. Alcoholic patients tend to miss the appointments more compared to other diseased patients.
# 5. Most of the appointments were missed on Tuesday and Wednesday and surprisingly most appointments were booked on the same days of the week.
# 6. SMS-Received or Scholorship seem to have no effect on the appointments show or no-show.
# 7. Adults followed by Seniors missed most of the appointments.
# 8. From the models, Random Forest followed by Logistic regression work best for the data.

# ### End
