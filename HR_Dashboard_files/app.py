from flask import Flask, render_template, request
app = Flask(__name__)

import pandas as pd
import statistics

@app.route('/result',methods= ['POST', 'GET'])
def result():
    global result



    if request.method == 'GET':
        result = request.args.get('Question')
        return render_template("Result.html",result = salary(result))


def salary(result):

    df = pd.read_csv("C:/Users/dhann/PycharmProjects/Flask/core_dataset.csv")
    name = result.split(' ')

    # x = df.loc[df['Employee Name'] == name[0]]
    # sdict = x.to_dict(orient='records')
    x = df[df['Employee Name'].str.contains(name[0]) == True]
    sdict = x.to_dict(orient='records')
    y = df[df['Department'].str.contains(name[0]) == True]
    sdict2 = y.to_dict(orient='records')
    z = df[df['RaceDesc'].str.contains(name[0]) == True]
    sdict3 = z.to_dict(orient='records')

    if sdict:
        print('Structure is not empty.')
        leaves = sdict[0]['Leaves taken']
        Total_leaves = sdict[0]['Total leaves']
        zip = sdict[0]['Zip']
        d_emp = sdict[0]['Days Employed']
        empname = sdict[0]['Employee Name']
        payrate = sdict[0]['PayRate']
        dept = sdict[0]['Department']
        color = "#19474F"
        position = sdict[0]['Position']



        result_obj = {
            'leaves': leaves,
            'total_leaves': Total_leaves,
            'name': empname,
            'zip': zip,
            'dates_employed': d_emp,
            'payrate' : payrate,
            'dept' : dept,
            'color' : color,
            'position' : position


        }
        return (result_obj)

    elif sdict2:

        y = df[df['Department'].str.contains(name[0]) == True]
        sdict = y.to_dict(orient='records')
        Mean_Age = statistics.mean(y.Age)
        Mean_payrate = statistics.mean(y.PayRate)
        deptname = sdict[0]['Department']
        dept = sdict[0]['Department']
        color = "#79774D"
        position = sdict[0]['Position']





        result_obj = {
            # 'leaves': leaves,
            'total_leaves': Mean_Age,
            # 'name': name[0],
             'zip': 33620,
            # 'dates_employed': d_emp
            'name': deptname,
            'dept': dept,

            'payrate' : Mean_payrate,
            'color' :   color,
            'position' : position


        }
        return (result_obj)


    elif sdict3:

        z = df[df['RaceDesc'].str.contains(name[0]) == True]
        sdict = z.to_dict(orient='records')

        Mean_payrate = statistics.mean(z.PayRate)
        racename = sdict[0]['RaceDesc']
        draw = "draw"

        color = "#79474A"

        result_obj = {
            # 'leaves': leaves,
            # 'total_leaves': Mean_Age,
            # 'name': name[0],
            #  'zip': 33620,
            # 'dates_employed': d_emp
            'name': racename,
            # 'dept': dept,
            'payrate': Mean_payrate,
            'color': color,
            'draw' : draw

        }
        return (result_obj)


    else : return (null)

    # else:
    #     z = df[df['RaceDesc'].str.contains(name[0]) == True]
    #     sdict = z.to_dict(orient='records')

        # if sdict:
        #     print('Structure is not empty.')
        #
        # else:
        #     print(
        #         'The value you gave does not reflect any person or position in the organisation. Please check and try again!')



    # leaves = sdict[0]['Leaves taken']
    # Total_leaves = sdict[0]['Total leaves']
    # zip = sdict[0]['Zip']
    # d_emp = sdict[0]['Days Employed']
    # result_obj = {
    #           'leaves': leaves,
    #           'total_leaves': Total_leaves,
    #           'name': name[0],
    #           'zip': zip,
    #            'dates_employed': d_emp
    # }
    # return (result_obj)



@app.route('/')
def index():
    return render_template("Vacation.html")

if __name__ == '__main__':
    app.run(debug = True)