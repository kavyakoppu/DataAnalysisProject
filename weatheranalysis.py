#!/usr/bin/env python
from __future__ import print_function
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

sparkcontext = SparkContext.getOrCreate() #appName='weatheranalysis'
sqlc = SQLContext(sparkcontext)

available_years = range(2000, 2020)  # all data from 2000-2019


def mkdataframe(filename):
    """
    Read filename and return a handle to a PySpark DataFrame
    """
    raw = sparkcontext.textFile(filename)
    data = raw.map(lambda x: x.split(','))
    table = data.map(lambda r: Row(station=r[0], date=r[1], minormax=r[2], degrees=int(r[3]), mflag=r[4], qflag=r[5], sflag=r[6], obstime=r[7]))
    df = sqlc.createDataFrame(table)
    return df.filter(df.qflag=='')  # not considering data with quality problems


def analyse_year_wise(years=available_years):
    """
    Analyse individual years in sequence
    """
    import pyspark.sql.functions as sqlfunc

    if not type(years) is list:
        years = [years]

    for year in years:

        df = mkdataframe('/user/tatavag/weather/%s.csv' % year)
        print("\n%s\n----\n" % year)
        
        # Average TMIN
        r = df.filter(df.minormax=='TMIN').groupBy().avg('degrees').first()
        print('Average TMIN : %f' % (r['avg(degrees)']))

        # Average TMAX
        r = df.filter(df.minormax=='TMAX').groupBy().avg('degrees').first()
        print('Average TMAX : %f' % (r['avg(degrees)']))
        
        # Lowest TMIN
        r = df.filter(df.minormax=='TMIN').groupBy().min('degrees').first()
        print('Min TMIN : %f' % (r['min(degrees)']))

        # Highest TMAX
        r = df.filter(df.minormax=='TMAX').groupBy().max('degrees').first()
        print('Max TMAX : %f' % (r['max(degrees)']))
                
        # Five hottest stations
        fivehot = df.filter(df.minormax=='TMAX').groupBy(df.station).agg(sqlfunc.max('degrees')).sort(sqlfunc.desc('max(degrees)')).limit(5).collect()
        print("5 Hottest stations :")
        i = 1
        for s in fivehot:
            t = float(s['max(degrees)'])
            print('\t%s : %f' % (s.station, t))
            i = i + 1

        # Five coldest stations (on average)
        fivecold = df.filter(df.minormax=='TMIN').groupBy(df.station).agg(sqlfunc.min('degrees')).sort(sqlfunc.asc('min(degrees)')).limit(5).collect()
        print("5 Coldest stations : \n")
        i = 1
        for s in fivecold:
            t = float(s['min(degrees)'])
            print('\t%s : %f' % (s.station, t))
            i = i + 1

        # Median TMIN
        r = df.filter(df.minormax=='TMIN').approxQuantile('degrees',[0.5], 0.25)
        print('Median TMIN : %f' % (r[0]))
        
        # Median TMAX
        r = df.filter(df.minormax=='TMAX').approxQuantile('degrees',[0.5], 0.25)
        print('Median TMAX : %f' % (r[0]))


def analyse_entire_dataset():
    """
    Analyse the entire dataset (2000-2019)
    """
    import pyspark.sql.functions as sqlfunc
    # from datetime import datetime as dt

    # Hottest and coldest day and corresponding weather stations in the entire dataset
    # Loading all datasets into a single DataFrame.

    print("\n-------------------------\n")
    df = mkdataframe('/user/tatavag/weather/20??.csv')

 
    # Coldest station
    coldest = df.filter(df.minormax=='TMIN').groupBy('station', 'date').min('degrees') \
                .sort(sqlfunc.asc('min(degrees)')).first()
    # date = dt.strptime(coldest.date, '%Y%m%d').strftime('%d %b %Y')
    print('Coldest station was %s on %s: %f'
          % (coldest.station, coldest.date, float(coldest['min(degrees)'])))

    # Hottest station
    hottest = df.filter(df.minormax=='TMAX').groupBy('station', 'date').max('degrees') \
                .sort(sqlfunc.desc('max(degrees)')).first()
    # date = dt.strptime(hottest.date, '%Y%m%d').strftime('%d %b %Y')
    print('Hottest station was %s on %s: %f'
          % (hottest.station, hottest.date, float(hottest['max(degrees)'])))

    # Median TMIN
    TMINmed = df.filter(df.minormax=='TMIN').approxQuantile('degrees',[0.5], 0.25)
    print('Median TMIN for the entire dataset: %f' % (TMINmed[0]))

    # Median TMAX
    TMAXmed = df.filter(df.minormax=='TMAX').approxQuantile('degrees',[0.5], 0.25)
    print('Median TMAX for the entire dataset: %f' % (TMAXmed[0]))

if __name__ == '__main__':
    analyse_year_wise()
    analyse_entire_dataset()
