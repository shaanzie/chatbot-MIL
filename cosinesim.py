import sqlite3

import spacy
nlp = spacy.load('en_core_web_md')

conn=sqlite3.connect('prevcontext.db')
cur=conn.cursor()
print("Connection successful!")



#cur.execute("""CREATE TABLE PREV (input VARCHAR(20))""")

threshold=0.8
i=0
lines=4 #minimum number of lines to understand context
inp=input("User: ")

exit = 0
while(inp!='Bye' and exit!=1):


        if i>lines:
            cur.execute("SELECT * from PREV")
            rows = cur.fetchall()

            for row in rows:
                #print(row[0])
                if(nlp(inp).similarity(nlp(row[0]))>=threshold):
                    print(nlp(inp).similarity(nlp(row[0])))
                    print("Done")
                    exit=1

        #Insert into database after checking, else checks with the same line (which is now at the end), and has cosine similarity=1
        #Can't insert nlp(inp) since the data type has been mentioned as string (so can't store nlp object)
        cur.execute("INSERT INTO PREV (input) values (?)", (inp,))
        conn.commit()
        i+=1

        if(exit!=1):
            inp=input("User:")
