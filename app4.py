import flask
from flask import request, jsonify
import math
import json
import requests
import pandas as pd
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.modules.elmo import Elmo, batch_to_ids
import warnings
from collections import defaultdict
from allennlp import predictors
from allennlp.predictors import Predictor
import spacy
spacy.load('en_core_web_sm')
import re
import math
import json
import requests
import pandas as pd
from collections import defaultdict

app = flask.Flask(__name__)
app.config["DEBUG"] = True

import pyodbc 
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=azgovprdsql11;'
                      'Database=OneFinReports;'
                      'Trusted_Connection=yes;')
cursor = conn.cursor()



def emailmatch(tkt):    
    sp1 = """EXEC [dbo].[SPVref] @TicketNumber=?"""
    params=tkt
    cursor.execute(sp1,params)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    email_df=pd.DataFrame(result_dicts)
    #converted=email_df.to_json(orient='records')
    return email_df
def regex(tkt):    
    sp3="""EXEC [dbo].[SPCRM]  @TicketNumber=?"""
    sp4="""EXEC [dbo].[SPPOVref_hs_v2] @Inv=?"""
    params=tkt
    cursor.execute(sp3,params)   
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    regex_df=pd.DataFrame(result_dicts)
    #print(regex_df)
    masterpodict=defaultdict(list)
    po_ref = pd.DataFrame()
    for i,row in regex_df.iterrows():
        po=re.findall('\W[0]*[9][8]\d{6}\W|\W[0]*[9][9]\d{6}\W|\W[0]*[9][7]\d{6}\W|\W[0]*[8][0]\d{6}\W|\W[7][0]\d{6}\W|\W[7][1]\d{6}\W|\W[1][0]\d{6}\W|\W[4][7]\d{7}\W|\W[4][1]\d{8}\W|\W[6][1]\d{8}\W|\W[6][0]\d{8}\W',row['description'])
        
        for p in po:
            if(p is not None):
                p1=re.sub('\W+','', p) 
                #print(p1)
                cursor.execute(sp4,p1) 
                column_names_list = [x[0] for x in cursor.description]
                result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
                po_ref=pd.DataFrame(result_dicts)
    return po_ref
def bidaf(tkt):    
    sp3="""EXEC [dbo].[SPCRM]  @TicketNumber=?"""
    sp5="""EXEC [dbo].[SPInvVref_hs_v2]  @Inv=?"""
    params=tkt
    cursor.execute(sp3,params)   
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    bidaf_df=pd.DataFrame(result_dicts)
    masterinv=defaultdict(list)
    model = Predictor.from_path("bidaf.tar.gz")
    question1 = "what is the invoice number?"
    #question2 = "what is the customer number?"
    #question3 = "what is the reference number?"
    invref = pd.DataFrame()
    for i,row in bidaf_df.iterrows():
        #print(row)
        inv_no1=model.predict(question1, row['description'])["best_span_str"]
        #print(inv_no1)
        #inv_no2=model.predict(question2,row['description'])["best_span_str"]
        #inv_no3=model.predict(question3,row['description'])["best_span_str"]
        cursor.execute(sp5,inv_no1) 
        column_names_list = [x[0] for x in cursor.description]
        result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
        #print(result_dicts)
        invref=pd.DataFrame(result_dicts)
    return invref
def html(tkt):
    f="S:\\Savitha\\html\\" 
    #f="C:\\Users\\v-savrav\\Text\\data\\html\\"
    sp6="""EXEC [dbo].[SPInvVref_hs_v2]  @Inv=?"""
    #sp7="""EXEC [dbo].[SPPOVref_hs_v2]  @Inv=?"""
    invlist=['MS Invoice Number','Invoice No.','Invoice #','Invoice Number','Invoice No','Vendor Invoice Number','Invoice#','Invoice reference',
            'Tax invoice number','MS Doc #','Invoice No.','Invoice','Microsoft Invoice', 'Invoice reference','INVOICE#','Document Number',
           'Invoice NO. or Descriptions','Reference Number','Reference','REFERENCE','Microsoft reference']
    polist=['PO Number','Master PO No.','Po #','ECIF PO NO.']
    html= pd.DataFrame()
    try:
        doc=f+tkt+'.html'
        df1=pd.read_html(doc,header= 0,flavor=['lxml', 'bs4'])
        #print(df1.columns)
        if(len(df1)>0):
            #print(len(df1))
            df =df1[-1]
            df_ll=[c for c in df.columns if c in invlist] 
            df_po=[c for c in df.columns if c in polist]  
            print(1)
            if(len(df_ll)>=1):
                for i in df[df_ll].values:
                    inv=str(i)
                    cursor.execute(sp6,inv) 
                    column_names_list = [x[0] for x in cursor.description]
                    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
                    print(result_dicts)
                    html=pd.DataFrame(result_dicts)
                    html['stage']='html'
                    print(html)
            elif(len(df_po)>=1):
                 for i in df[df_po].values:
                    po=str(i)
                    cursor.execute(sp7,inv) 
                    column_names_list = [x[0] for x in cursor.description]
                    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
                    #print(result_dicts)
                    html=pd.DataFrame(result_dicts)
                    html['stage']='html' 
                    print(html)
            else:
                print("no results") 
        return html               
    except ValueError:
        pass  

@app.route('/getVendor', methods=['GET'])
def api():
    tkt = str(request.args['tkt'])
    print(tkt)
    final=pd.DataFrame()
    final['tkt']=tkt
    df1=emailmatch(tkt)
    if (df1.shape[0]>=1):
        final=df1
    elif(df1.shape[0]<=0):
        df2=regex(tkt)
        df2['stage']='regex'
        if(df2.shape[0]>=1):
            final=df2
        else:
            df3=bidaf(tkt)
            df3['stage']='bidaf'
            if(df3.shape[0]>=1):
                final=df3
            else:
                df4=html(tkt)
                if(df4 is not None):
                    final=df4
                    final['stage']='html'
                else:
                    final['stage']="no results"
    convert=final.to_json(orient='records')
    print("convert",convert)
    return convert

@app.route('/ageing', methods=['GET']) 
def ageing():
    tkt=str(request.args['tkt'])
    sp2 = """EXEC [dbo].[usp_getAgeing] @TicketNumber=?"""
    params=tkt
    cursor.execute(sp2,params)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    converted=df.to_json(orient='records')
    return converted

@app.route('/RMAP', methods=['GET']) 
def RMAP(): 
	tkt=str(request.args['tkt'])
	sp1 = """EXEC [dbo].[usp_getRMAP14] @TicketNumber=?"""
	params=tkt
	cursor.execute(sp1,params)
	column_names_list = [x[0] for x in cursor.description]
	result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
	df=pd.DataFrame(result_dicts)
	converted=df.to_json(orient='records')
	return converted

@app.route('/All_PO', methods=['GET']) 
def all_po():
	#ven='0002280529'
    ven=str(request.args['ven'])
    sp4 = """EXEC [dbo].[uspGetAllPOs] @VendorNumber=?"""
    params=ven
    cursor.execute(sp4,params)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    converted=df.to_json(orient='records')
    return converted

@app.route('/api4', methods=['GET']) 
def api4():
    ven=str(request.args['ven_no'])
    cc=str(request.args['cc'])
    sp5 = """EXEC [dbo].uspGetAllTicketsv2 @VendorNumber=?,@CompanyCode=?"""
    #params= ven
    cursor.execute(sp5,ven,cc)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    #df['Statecode']='0'
    #df['dateCreated']='20-nov-2019'
    #df['Tickettype']='support'
    converted=df.to_json(orient='records')
    return converted

@app.route('/api3', methods=['GET']) 
def api3():
    ven_no=str(request.args['ven_no'])
    po=str(request.args['po'])    
    sp2 = """EXEC [dbo].[SPVRef_Invoice] @VendorNumber=? , @PONumber=?"""
    #params=tkt
    cursor.execute(sp2,ven_no,po)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    converted=df.to_json(orient='records')
    return converted

@app.route('/api2', methods=['GET'])
def api2():
    ven=str(request.args['ven'])
    sp4 = """EXEC [dbo].[SPVRef_ZOPINV]  @VendorNumber=?"""
    params=ven
    cursor.execute(sp4,params)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    converted=df.to_json(orient='records')
    return converted
@app.route('/api1', methods=['GET'])
def api1():
    tkt=str(request.args['tkt'])
    sp4 = """EXEC [dbo].[SPVRef_Ticket]    @TicketNumber=?"""
    params=tkt
    cursor.execute(sp4,params)
    column_names_list = [x[0] for x in cursor.description]
    result_dicts = [dict(zip(column_names_list, row)) for row in cursor.fetchall()]
    df=pd.DataFrame(result_dicts)
    converted=df.to_json(orient='records')
    return converted
app.run(host='10.144.252.76')
conn.commit()