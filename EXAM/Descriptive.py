import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


########################################## loading data ##########################################

data_path = 'C:/Users/Munth/Desktop/Social_data_science_exam/data/'

data = pd.read_csv(data_path + 'user_data_public.csv')
questions  = pd.read_csv(data_path + 'question_data.csv',sep = ';')

########################################## Manipulating data ##########################################

data["Questions_answered"] = pd.notnull(data).sum(axis = 1)
data["gender"] = data["gender"].fillna(value = "Other")
data["gender"].unique()

table1 =pd.DataFrame({'Sex':[],'Observations':[],'Straight_ratio':[],'Avg questions answered':[],'Religious':[]})


questions["Keywords"] = questions["Keywords"].fillna(value = "Other")
questions_banned = list(questions.loc[questions.loc[:,"N"] <= 20000,"Unnamed: 0"])
questions_less = questions.loc[questions.loc[:,"N"] > 20000,:].reset_index()

data_reduced = data.drop(questions_banned, axis = 1)
data_reduced["Questions_answered"] = pd.notnull(data_reduced).sum(axis = 1)
questions_answered_reduced = data_reduced["Questions_answered"]
data_reduced = data_reduced.loc[data_reduced.loc[:,"Questions_answered"] >= 400,:]
data_reduced = data_reduced.reset_index().drop(["index"],axis = 1)

# Creating data for questions graph

categories = ['sex/intimacy','cognitive','descriptive','BDSM','preference',
              'opinion','technology','politics','religion/superstition', 'Other']
groups = {'Category': categories,'Value':[0]*len(categories)}

groups_df1 = pd.DataFrame(data = groups)
for i in range(0,len(questions["Keywords"])) :
    tags = questions["Keywords"][i].split(';')
    for j in range(0,len(tags)) :
        groups_df1.loc[groups_df1.loc[:,"Category"] == tags[j].strip(),"Value"] = groups_df1.loc[groups_df1.loc[:,"Category"] == tags[j].strip(),"Value"] + 1
groups_df1 = groups_df1.sort_values(by = "Value").reset_index().drop(["index"],axis = 1)


groups_df2 = pd.DataFrame(data = groups)
for i in range(0,len(questions_less["Keywords"])) :
    tags = questions_less["Keywords"][i].split(';')
    for j in range(0,len(tags)) :
        groups_df2.loc[groups_df2.loc[:,"Category"] == tags[j].strip(),"Value"] = groups_df2.loc[groups_df2.loc[:,"Category"] == tags[j].strip(),"Value"] + 1
groups_df2 = groups_df2.sort_values(by = "Category").reset_index().drop(["index"],axis = 1)

    
groups_df1["Data cleaning"] = "Before"
groups_df2["Data cleaning"] = "After"

groups = groups_df1.append(groups_df2)

groups['Category'] = pd.Categorical(groups['Category'], ['BDSM','technology', 'cognitive','Other',
                     'religion/superstition','politics','sex/intimacy','opinion','preference',
                     'descriptive'])
groups_df2['Category'] = pd.Categorical(groups_df2['Category'], ['BDSM','technology', 'cognitive','Other',
                     'religion/superstition','politics','sex/intimacy','opinion','preference',
                     'descriptive'])
    
groups = groups.sort_values(['Category','Data cleaning'], ascending=[1,0]).reset_index().drop(["index"],axis = 1)
groups_df2 = groups_df2.sort_values(['Category','Data cleaning'], ascending=[1,0]).reset_index().drop(["index"],axis = 1)
    
### prepping data for stacked columns chart

# prepping gender orientation
gender_orientation = data_reduced["gender_orientation"].value_counts()
gender_orientation = pd.DataFrame(data = {'Gender_orientation':list(gender_orientation.index),'Value':list(gender_orientation),'Gender':['empty'] *len(gender_orientation),'Orientation':['empty']*len(gender_orientation)})
for i in range(0,len(gender_orientation)):
    tags = gender_orientation['Gender_orientation'][i].split('_')
    gender_orientation["Orientation"][i] = tags[0] 
    gender_orientation["Gender"][i] = tags[1]
gender_orientation = gender_orientation.pivot(index = 'Orientation',columns = 'Gender', values = 'Value')

plt.style.use(u'ggplot')
fig = gender_orientation.loc[:,['female','male']].plot.bar(stacked=True,grid = False, figsize=(13,7),
                            #title = 'Distribution of sexuality across gender',
                            fontsize = 12,rot = 0)
fig.set(xlabel='Sexuality', ylabel='Number of respondants')
fig = fig.get_figure()
fig.gca().yaxis.grid(True)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Orientation.png")

# Prepping age and gender
age = data_reduced.loc[~data_reduced["d_age"].isnull(),:].reset_index()
age = age[["d_age","gender"]]
age["Age_bucket"] = ['empty'] * len(age["d_age"])
age.loc[age["d_age"] < 25,"Age_bucket"] = "24 or less"
age.loc[(age["d_age"] >= 25) & (age["d_age"] <= 29),"Age_bucket"] = "25-29"
age.loc[(age["d_age"] >= 30) & (age["d_age"] <= 34),"Age_bucket"] = "30-34"
age.loc[(age["d_age"] >= 35) & (age["d_age"] <= 39),"Age_bucket"] = "35-39"
age.loc[(age["d_age"] >= 40) & (age["d_age"] <= 44),"Age_bucket"] = "40-44"
age.loc[(age["d_age"] >= 45) & (age["d_age"] <= 49),"Age_bucket"] = "45-49"
age.loc[(age["d_age"] >= 50) & (age["d_age"] <= 54),"Age_bucket"] = "50-54"
age.loc[(age["d_age"] >= 55) & (age["d_age"] <= 59),"Age_bucket"] = "55-59"
age.loc[age["d_age"] >= 60,"Age_bucket"] = "60+"
age = age.groupby(['gender','Age_bucket'], as_index=False).agg({'d_age':'count'})
age = age.pivot(index = 'Age_bucket',columns = 'gender', values = 'd_age')
age = age.fillna(value = 0)
age.rename(index={"24 or less":"<25"},inplace = True) 

fig = age.loc[:,['Woman','Man']].plot(kind = 'bar',grid = False,stacked=True, figsize=(13,7),
             #title = 'Distribution of age across gender',
             fontsize = 12,rot = 0)
fig.set(xlabel='Age buckets', ylabel='Number of respondants')
fig = fig.get_figure()
fig.gca().yaxis.grid(True)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Age.png")
    
# Prepping data for religion
religion = data_reduced[['d_religion_type','gender']]
religion.loc[data_reduced["d_religion_type"] == '-','d_religion_type'] = 'Missing'
religion['Value'] = [1] * len(religion['gender'])
religion = religion.groupby(['gender','d_religion_type'], as_index=False).agg({'Value':'count'})
religion = religion.pivot(index = 'd_religion_type',columns = 'gender', values = 'Value').fillna(value = 0)
religion['Total'] = religion['Man'] + religion['Woman'] + religion['Other']
religion = religion.sort_values(by = "Total")

fig = religion.loc[:,['Woman','Man']].plot(kind = 'bar',grid = False,stacked=True, figsize=(13,7),
             fontsize = 12,rot = 0)
fig.set(xlabel='Religion', ylabel='Number of respondants')
fig = fig.get_figure()
fig.gca().yaxis.grid(True)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Religion.png")

data_reduced.loc[(data_reduced["d_religion_type"] != "Agnosticism") & (data_reduced["d_religion_type"] != "Atheism") & (data_reduced["gender"] == "Woman") ,"q41"].value_counts()
data_reduced.loc[(data_reduced["d_religion_type"] != "Agnosticism") & (data_reduced["d_religion_type"] != "Atheism") & (data_reduced["gender"] == "Man") ,"q41"].value_counts()

# Prepping random data
random = data_reduced[['gender','d_smokes','d_drugs','d_offspring_current',"d_education_type","d_bodytype",]]

ra = pd.DataFrame({'Categories':['Smokers','Do_drugs','Have_kids','University_degree','Fit/athletic','Curvy/overweight','Average_body'],
      'Value':[random.loc[data_reduced["d_smokes"] != "No",'d_smokes'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[data_reduced["d_drugs"] != "Never",'d_drugs'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[data_reduced["d_offspring_current"] == "kids",'d_offspring_current'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[(data_reduced["d_education_type"] == "university") | (data_reduced["d_education_type"] == "masters program"),'d_education_type'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[(data_reduced["d_bodytype"] == "Fit") | (data_reduced["d_bodytype"] == "Athletic"),'d_bodytype'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[(data_reduced["d_bodytype"] == "Curvy") | (data_reduced["d_bodytype"] == "Overweight"),'d_bodytype'].value_counts().sum()/len(random[["gender"]])*100,
               random.loc[data_reduced["d_bodytype"] == "Average",'d_bodytype'].value_counts().sum()/len(random[["gender"]])*100
               ]})
ra = ra.set_index(ra["Categories"],'Categories') 
    
fig = ra.plot(kind = 'bar',grid = False, legend = False, figsize=(13,7),
             fontsize = 12,rot = 0,color=[plt.cm.Paired(np.arange(len(ra)))])
fig.set(xlabel='Categories', ylabel='Percentage of respondants')
fig = fig.get_figure()
fig.gca().yaxis.grid(True)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Random.png")
 
# Deleting unused variables
del  tags, i,j,categories,questions_banned

########################################## more graphs ##########################################


fig, ax = plt.subplots(figsize=(13,6), ncols=1, nrows=1)
plt.style.use(u'ggplot')
h_num_answered_question = sns.distplot(questions["N"], kde = False, bins = 30,axlabel = "Number of answers",
                                       hist_kws={"rwidth":0.75,'edgecolor':'black', 'alpha':1.0}).tick_params(labelsize=20)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Figure1a.png")

fig, ax = plt.subplots(figsize=(13,6), ncols=1, nrows=1)
h_questions_answered = sns.distplot(questions_answered_reduced, kde = False, bins = 50,#axlabel = "Number of questions answered",
                                    hist_kws={"rwidth":0.75,'edgecolor':'black', 'alpha':1.0}).tick_params(labelsize=20)                                
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Figure1b.png")

fig, ax = plt.subplots(figsize=(13,7), ncols=1, nrows=1,)
h_question_categories=sns.barplot(x='Category',y='Value',hue = "Data cleaning",data=groups)
h_question_categories.set(xlabel='Question category', ylabel='Number of questions')
for index, row in groups_df2.iterrows():
    h_question_categories.text(row.name,row.Value, row.Value)
fig.savefig("C:/Users/Munth/Desktop/Social_data_science_exam/data/Images/Questions.png")
