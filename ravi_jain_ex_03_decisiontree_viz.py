import pydotplus
import graphviz
import collections
import pandas as pd
import pandas_profiling

from sklearn import tree

# import the dataset
df_diabetes = pd.read_csv('diabetes.csv')

# creating and saving data exploration report
# print('\n---Starting the data exploration for diabetes dataset\n')
# df_diabetes.profile_report()
# profile = df_diabetes.profile_report(title='Diabetes_Profiling_report')
# profile.to_file(output_file='Profiling_report.html')
# print('\n---End of analysis\n')

# transform dataframe into a list
diabetes_list = df_diabetes.values.tolist()

# Get all outcomes as a list (Y)
outcome_list = df_diabetes['Outcome'].values.tolist()

# Get all patient info except for outcome as a dataframe
df_patient_info = df_diabetes[df_diabetes.columns[:-1]]

# Transform patient info into a list
patient_info_list = df_patient_info.values.tolist()

# data feature names for patient info
attr_names = df_patient_info.columns.tolist()

# training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(patient_info_list, outcome_list)

list_of_predictions = []
for input in patient_info_list:
    prediction = clf.predict([input])
    print('\nPredicting outcome for a subject with the following characteristics:')
    print('     ', attr_names[0],'=', input[0], ', ',attr_names[1],'=',input[1], ', ',attr_names[2],'=',input[2],
          ', ',attr_names[3],'=',input[3], ', ',attr_names[4],'=',input[4], ', ',attr_names[5],'=',input[5], ', ',
          attr_names[6],'=',input[6], ', ',attr_names[7],'=',input[7],)
    print('\nThe outcome is: ', prediction[0])
    list_of_predictions.append(prediction[0])

count_right = 0
for index, prediction in enumerate(list_of_predictions):
    if prediction == outcome_list[index]:
        count_right += 1

print('\nThe Accuracy of the classifier is: ', count_right/len(outcome_list))

# dot_data = tree.export_graphviz(clf,
#                                 feature_names=attr_names,
#                                 out_file=None,
#                                 filled=True,
#                                 rounded=True)


# graph = pydotplus.graph_from_dot_data(dot_data)
# edges = collections.defaultdict(list)
#
# for edge in graph.get_edge_list():
#     edges[edge.get_source()].append(int(edge.get_destination()))
#
# for edge in edges:
#     edges[edge].sort()
#     for i in range(2):
#         dest = graph.get_node(str(edges[edge][i]))[0]
#
# graph.write_png('tree.png')



