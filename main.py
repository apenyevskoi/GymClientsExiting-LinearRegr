import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def importcsv(path):
    df = pd.read_csv( path )
    df = df.iloc[:, 1:len( df.columns ) - 1]
    countOk = 0
    for i in ~pd.isna( df ).all( ):
        if i == True:
            countOk += 1
        else:
            return 'NA\'s in DF'
    if countOk == len( df.columns ):
        return df

def f1ScoreTest(df, gaugedf):
    x_train, x_test, y_train, y_test = train_test_split( df[['zipcode', 'age', 'partner_company', 'friend_promo',
                                                             'contract_period', 'lifetime', 'class_registration_weekly',
                                                             'avg_additional_charges_total', 'cancellation_freq']],
                                                         df['exited'],
                                                         test_size=1 )
    model = LinearRegression( )
    model.fit( x_train, y_train )
    predictionTrain = model.predict( df[['zipcode', 'age', 'partner_company', 'friend_promo',
                                         'contract_period', 'lifetime', 'class_registration_weekly',
                                         'avg_additional_charges_total', 'cancellation_freq']] )

    # print(sklearn.metrics.f1_score( df['exited'], [0 if x <= 0.51 else 1 for x in predictionTrain]))
    #
    for i in [ x/100 for x in range(0,100,5)]:
        print(i, sklearn.metrics.f1_score( df['exited'], [0 if x <= i else 1 for x in predictionTrain]))


def main2():

    df = importcsv( 'gym_data.csv' )
    df.columns = ['registration', 'zipcode', 'age', 'partner_company', 'friend_promo',
       'contract_period', 'lifetime', 'class_registration_weekly',
       'avg_additional_charges_total', 'cancellation_freq', 'exited']
    gaugedf = importcsv( 'gym_test.csv' )
    gaugedf.columns = ['registration', 'zipcode', 'age', 'partner_company', 'friend_promo',
                       'contract_period', 'lifetime', 'class_registration_weekly',
                       'avg_additional_charges_total', 'cancellation_freq']

    pd.set_option( 'display.max_columns', None )
    pd.set_option( 'display.max_rows', None )

    x_train, x_test, y_train, y_test = train_test_split( df[['zipcode', 'age', 'partner_company', 'friend_promo',
                                                             'contract_period', 'lifetime', 'class_registration_weekly',
                                                             'avg_additional_charges_total', 'cancellation_freq']],
                                                         df['exited'],
                                                         test_size=1)
    model = LinearRegression( )
    model.fit( x_train, y_train )
    print(
        pd.DataFrame(model.coef_, ['zipcode', 'age', 'partner_company', 'friend_promo',
                                                             'contract_period', 'lifetime', 'class_registration_weekly',
                                                             'avg_additional_charges_total', 'cancellation_freq'])
    )

    predictionForF1 = model.predict(df[['zipcode', 'age', 'partner_company', 'friend_promo',
                                                             'contract_period', 'lifetime', 'class_registration_weekly',
                                                             'avg_additional_charges_total', 'cancellation_freq']])
    prediction = model.predict( gaugedf[['zipcode', 'age', 'partner_company', 'friend_promo',
                                         'contract_period', 'lifetime', 'class_registration_weekly',
                                         'avg_additional_charges_total', 'cancellation_freq']] )
    prediction = pd.DataFrame( {'predictionregr': prediction} )
    gaugedf['predictionregr'] = prediction
    gaugedf = gaugedf.assign( exited=gaugedf['predictionregr'].apply( lambda x: 0 if x <= 0.51 else 1 ) )
    gaugedf = gaugedf.drop('predictionregr', axis=1)

    #check F1 score
    # f1ScoreTest( df, gaugedf )

    gaugedf.columns = ['Registration', 'Zipcode', 'Age', 'Partner_company', 'Friend_promo',
       'Contract_period', 'Lifetime', 'Class_registration_weekly',
       'Avg_additional_charges_total', 'Cancellation_freq', 'Exited']
    print('Ratio of exited clients to all clients in training data (gym_data.csv):',
          df['age'].loc[(df['exited'] == 1)].count() / len(df))
    print('Ratio of exited clients to all clients in test data (gym_test.csv):',
          gaugedf['Age'].loc[gaugedf['Exited'] == 1].count( ) / len( gaugedf ))
    print('F1 score in training data (gym_data.csv):',
          sklearn.metrics.f1_score(df['exited'], [ 0 if x <= 0.51 else 1 for x in predictionForF1 ]))
    
    

    #gaugedf.to_csv(file_name, sep='\t')

def outputGauge(out):
    print(out)

if __name__ == '__main__':
    main2( )


''''
# MANUAL TRAINING
def ageDistribution(df):
    sns.set_theme( )
    sns.displot( df, x='age', hue='exited', kind='kde', fill=True, palette=sns.color_palette( 'bright' )[1:3], height=5,
                 aspect=1.5 )
    plt.show( )


def zipcodeProbability(df):
    zip, exited, zipprob = [], [], []
    for i in df['zipcode'].unique( ):
        recordsCount = df['zipcode'].loc[(df['zipcode'] == i)].count( )
        for j in [0,1]:
            z = df['zipcode'].loc[(df['zipcode'] == i) &
                                  (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            if z == 0:
                z = 0.0001
            zipprob.append( z )
            exited.append(j)
            zip.append( i )
    zipdf = pd.DataFrame(
        {'zipcode': zip,
         'exited':exited,
         'zipprob': zipprob}
    )
    return zipdf


def ageProbability(df):
    age, exited, ageprob = [], [], []
    for i in df['age'].sort_values( ).unique( ):
        recordsCount = df['age'].loc[df['age'] == i].count( )
        for j in [0, 1]:
            z = df['age'].loc[(df['age'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            ageprob.append( z )
            exited.append( j )
            age.append( i )
    agedf = pd.DataFrame(
        {'age': age,
         'exited': exited,
         'ageprob': ageprob}
    )
    return agedf


def partnerProbability(df):
    partner, exited, partnerprob = [], [], []
    for i in df['partner_company'].sort_values( ).unique( ):
        recordsCount = df['partner_company'].loc[df['partner_company'] == i].count( )
        for j in [0, 1]:
            z = df['partner_company'].loc[(df['partner_company'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            partnerprob.append( z )
            exited.append( j )
            partner.append( i )
    partnerdf = pd.DataFrame(
        {'partner_company': partner,
         'exited': exited,
         'partnerprob': partnerprob}
    )
    return partnerdf


def friendProbability(df):
    friend, exited, friendprob = [], [], []
    for i in df['friend_promo'].sort_values( ).unique( ):
        recordsCount = df['friend_promo'].loc[df['friend_promo'] == i].count( )
        for j in [0, 1]:
            z = df['friend_promo'].loc[(df['friend_promo'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            friendprob.append( z )
            exited.append( j )
            friend.append( i )
    frienddf = pd.DataFrame(
        {'friend_promo': friend,
         'exited': exited,
         'friendprob': friendprob}
    )
    return frienddf


def contractPeriodProbability(df):
    contract, exited, contractProb = [], [], []
    for i in [1, 6, 12]:
        recordsCount = df['contract_period'].loc[df['contract_period'] == i].count( )
        for j in [0, 1]:
            z = df['contract_period'].loc[(df['contract_period'] == i) & (df['exited'] == j)].count( ) / recordsCount
            contractProb.append( z )
            exited.append( j )
            contract.append( i )
    contractdf = pd.DataFrame(
        {'contract_period': contract,
         'exited': exited,
         'contractprob': contractProb}
    )
    return contractdf


def lifetimeProbability( df ):
    lifetime, exited, lifetimeProb = [], [], []
    for i in df['lifetime'].unique():
        recordsCount = df['lifetime'].loc[df['lifetime'] == i].count( )
        for j in [0, 1]:
            z = df['lifetime'].loc[(df['lifetime'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            if z == 0:
                z = 0.01
            lifetimeProb.append( z )
            exited.append( j )
            lifetime.append( i )
    lifetimedf = pd.DataFrame(
        {'lifetime': lifetime,
         'exited': exited,
         'lifetimeprob': lifetimeProb}
    )
    return lifetimedf

def classRegistrationWeeklyProbability( df ):
    maxReg = np.floor(df['class_registration_weekly'].max())
    ranges = df['class_registration_weekly'].groupby(pd.cut(df['class_registration_weekly'], np.append(np.arange(-0.004,maxReg,1), df['class_registration_weekly'].max()))).count()
    df = df.assign(tmprange = pd.cut(df['class_registration_weekly'], np.append(np.arange(-0.004,maxReg,1), df['class_registration_weekly'].max())))
    reg, exited, regProb = [], [], []
    for i in ranges.index.values:
        recordsCount = df['tmprange'].loc[df['tmprange'] == i].count( )
        for j in [0, 1]:
            z = df['tmprange'].loc[(df['tmprange'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            if z == 0:
                z = 0.01
            regProb.append( z )
            exited.append( j )
            reg.append( i )
    regdf = pd.DataFrame(
        {'tmprange': reg,
         'exited': exited,
         'regprob': regProb}
    )
    df = pd.merge(df, regdf, on=['tmprange', 'exited'])
    df = df.drop('tmprange', axis=1)
    return df, regdf

def cancelationFreqProbability( df ):
    df = df.assign(tmpcancellation = pd.cut(df['cancellation_freq'], np.arange(-0.001,1.001,0.25)))
    cancel, exited, cancelProb = [], [], []
    for i in df['tmpcancellation'].unique():
        recordsCount = df['tmpcancellation'].loc[df['tmpcancellation'] == i].count( )
        for j in [0, 1]:
            z = df['tmpcancellation'].loc[(df['tmpcancellation'] == i) & (df['exited'] == j)].count( ) / recordsCount
            if z == 1:
                z = 0.99
            if z == 0:
                z = 0.01
            cancelProb.append( z )
            exited.append( j )
            cancel.append( i )
    canceldf = pd.DataFrame(
        {'tmpcancellation': cancel,
         'exited': exited,
         'cancelprob': cancelProb}
    )
    df = pd.merge( df, canceldf, on=['tmpcancellation', 'exited'] )
    df = df.drop( 'tmpcancellation', axis=1 )
    return df, canceldf

def probabilitiesCalculation(df, zipdf, agedf, partnerdf, frienddf, contractdf, lifetimedf):
    df['zipprob'] = 'NA'
    df['ageprob'] = 'NA'
    df['partnerprob'] = 'NA'
    df['friendprob'] = 'NA'
    df['contractprob'] = 'NA'
    df['lifetimeprob'] = 'NA'
    conditions = [
        (df['zipcode'] == 57328) & (df['exited'] == 0),
        (df['zipcode'] == 57328) & (df['exited'] == 1),
        (df['zipcode'] == 29941) & (df['exited'] == 0),
        (df['zipcode'] == 29941) & (df['exited'] == 1),
        (df['zipcode'] == 33726) & (df['exited'] == 0),
        (df['zipcode'] == 33726) & (df['exited'] == 1),
        (df['zipcode'] == 65232) & (df['exited'] == 0),
        (df['zipcode'] == 65232) & (df['exited'] == 1)
    ]
    choices = [prob for prob in zipdf['zipprob']]
    df['zipprob'] = np.select( conditions, choices )

    #pd.set_option('display.max_columns', None)

    tmpdf = pd.merge(df, agedf, on=['age', 'exited'])
    df = tmpdf
    df['ageprob'] = df['ageprob_y']
    df = df.drop(['ageprob_y', 'ageprob_x'], axis=1)

    tmpdf = pd.merge(df, partnerdf, on=['partner_company', 'exited'])
    df = tmpdf
    df['partnerprob'] = df['partnerprob_y']
    df = df.drop(['partnerprob_y', 'partnerprob_x'], axis=1)

    tmpdf = pd.merge(df, frienddf, on=['friend_promo', 'exited'])
    df = tmpdf
    df['friendprob'] = df['friendprob_y']
    df = df.drop(['friendprob_y', 'friendprob_x'], axis=1)

    tmpdf = pd.merge( df, contractdf, on=['contract_period', 'exited'] )
    df = tmpdf
    df['contractprob'] = df['contractprob_y']
    df = df.drop( ['contractprob_y', 'contractprob_x'], axis=1 )

    tmpdf = pd.merge( df, lifetimedf, on=['lifetime', 'exited'] )
    df = tmpdf
    df['lifetimeprob'] = df['lifetimeprob_y']
    df = df.drop( ['lifetimeprob_y', 'lifetimeprob_x'], axis=1 )

    return df

def unionProbability(df):
    pd.set_option('display.max_columns', None)
    #pd.set_option( 'display.max_rows', None )
    df = df.assign(unionprob = np.sum(df[['zipprob',
                                          'ageprob',
                                          'partnerprob',
                                          'friendprob',
                                          'contractprob',
                                          'lifetimeprob',
                                          'regprob',
                                          'cancelprob']], axis=1))                                                         # P(ABCDE)=P(A)+...+P(E)-P(AB)-...-P(DE)+P(ABC)+...+P(CDE)-.....
    df['unionprob'] = df['unionprob'] - df[['zipprob', 'ageprob', 'partnerprob', 'friendprob','contractprob', \
                                          'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply(lambda r: np.sum( [ comb[0]*comb[1] for comb in list(combinations(r,2)) ] ),
                                        axis=1)                                                                         # r = 2 substraction
    df['unionprob'] = df['unionprob'] + df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                          'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply( lambda r: np.sum( [comb[0] * comb[1] * comb[2] for comb in list( combinations( r, 3 ) )] ),
                                       axis=1 )                                                                         # r = 3 addition
    df['unionprob'] = df['unionprob'] - df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                          'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply( lambda r: np.sum( [comb[0] * comb[1] * comb[2] * comb[3] for comb in list( combinations( r, 4 ) )] ),
                                       axis=1 )                                                                         # r = 4 substraction
    df['unionprob'] = df['unionprob'] + df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                            'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply( lambda r: np.sum( [comb[0] * comb[1] * comb[2] * comb[3] * comb[4] for comb in list( combinations( r, 5 ) )] ),
                                        axis=1 )                                                                        # r = 5 addition
    df['unionprob'] = df['unionprob'] - df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                            'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply( lambda r: np.sum( [comb[0] * comb[1] * comb[2] * comb[3] * comb[4] * comb[5] for comb in list( combinations( r, 6 ) )] ),
                                        axis=1 )                                                                        # r = 6 substraction
    df['unionprob'] = df['unionprob'] + df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                            'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply( lambda r: np.sum([comb[0] * comb[1] * comb[2] * comb[3] * comb[4] * comb[5] * comb[6] for comb in list( combinations( r, 7 ) )] ),
                                       axis=1 )                                                                         # r = 7 addition
    df['unionprob'] = df['unionprob'] - df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                          'lifetimeprob', 'regprob', 'cancelprob']]. \
                                apply(lambda r: np.prod(r) , axis=1)                                                    # end substraction


    print(df['unionprob'].describe())
    df = df.assign(test = df['unionprob'].apply(lambda p: 0 if p > 0.9999 else 1))
    #print(df)
    # print(df[['exited', 'unionprob', 'test']].iloc[6117:].query('test == 0').count())
    # print( df[['exited', 'unionprob', 'test']].iloc[:6118].query( 'test == 1' ) )
    #print(df[['exited', 'unionprob', 'test']].iloc[6100:])
    #print(df[['exited','unionprob', 'test']].iloc[:6100].query('unionprob < 0.993'))
'''

'''
def main2():

    # MANUAL TRAINING
    zipdf = zipcodeProbability( df )
    agedf = ageProbability( df )
    partnerdf = partnerProbability( df )
    frienddf = friendProbability( df )
    contractdf = contractPeriodProbability( df )
    lifetimedf = lifetimeProbability( df )
    df, regdf = classRegistrationWeeklyProbability( df )
    df, canceldf = cancelationFreqProbability( df )

    df = probabilitiesCalculation( df, zipdf, agedf, partnerdf, frienddf, contractdf, lifetimedf )
    df = df.assign(multiply = df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                          'lifetimeprob', 'regprob', 'cancelprob']].apply(lambda r: np.prod(r), axis=1))
    df = df.assign( sumcol=df[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                                 'lifetimeprob', 'regprob', 'cancelprob']].apply( lambda r: np.sum( r )/8, axis=1 ) )
    unionProbability(df)
 
    gaugedf = pd.merge(gaugedf, zipdf[zipdf['exited'] == 0].iloc[:,[0,2]], on=['zipcode'])
    gaugedf = pd.merge(gaugedf, agedf[agedf['exited'] == 0].iloc[:,[0,2]], on=['age'])
    gaugedf = pd.merge( gaugedf, partnerdf[partnerdf['exited'] == 0].iloc[:, [0, 2]], on=['partner_company'] )
    gaugedf = pd.merge( gaugedf, frienddf[frienddf['exited'] == 0].iloc[:, [0, 2]], on=['friend_promo'] )
    gaugedf = pd.merge( gaugedf, contractdf[contractdf['exited'] == 0].iloc[:, [0, 2]], on=['contract_period'] )
    gaugedf = pd.merge( gaugedf, lifetimedf[lifetimedf['exited'] == 0].iloc[:, [0, 2]], on=['lifetime'] )
    gaugedf = gaugedf.assign(tmprange = pd.cut(gaugedf['class_registration_weekly'],
                                               np.arange(-0.004,
                                                         np.ceil(gaugedf['class_registration_weekly'].max()),
                                                         1)))
    gaugedf = pd.merge(gaugedf, regdf[regdf['exited'] == 0].iloc[:, [0, 2]], on=['tmprange'])
    gaugedf = gaugedf.drop(['tmprange'], axis=1)
    gaugedf = gaugedf.assign( tmpcancellation=pd.cut( gaugedf['cancellation_freq'], np.arange(-0.001,1.001,0.25) ) )
    gaugedf = pd.merge( gaugedf, canceldf[canceldf['exited'] == 0].iloc[:, [0, 2]], on=['tmpcancellation'] )
    gaugedf = gaugedf.drop( ['tmpcancellation'], axis=1 )

    gaugedf = gaugedf.assign( sumcol=gaugedf[['zipprob', 'ageprob', 'partnerprob', 'friendprob', 'contractprob', \
                               'lifetimeprob', 'regprob', 'cancelprob']].apply( lambda r: np.sum( r ) / 8, axis=1 ) )
    unionProbability(gaugedf)
    gaugedf = gaugedf.assign( testexited = gaugedf['sumcol'].apply(lambda x: 0 if x >= 0.613323 else 1))
    print(gaugedf[gaugedf['testexited'] == 1])
    print(1983/8100, 8/200)
'''
