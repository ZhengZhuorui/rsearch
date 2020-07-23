import sqlite3
import numpy
import random
import os
import time
import shutil

import json

class DBConnector():
	def __init__(self, path = './Image1000.db'):
		self.conn = sqlite3.connect(path, check_same_thread = False)
		self.createTable()
		print("Open success in path:", path)

	def __del__(self):
		self.conn.close()

	def insert(self, idx, timestamp, lant, lont, feature, filename):
		c = self.conn.cursor()
		f = json.dumps(feature)
		c.execute('''insert into RemoteSensing(ID, time, lat, lng, feature, filename)
			values(?, ?, ?, ?, ?, ?);
			''', (idx, timestamp, lant, lont, f, filename))
		self.conn.commit()

	def createTable(self):
		c = self.conn.cursor()
		c.execute('''create table if not exists RemoteSensing(
				ID integer primary key autoincrement,
				time char[12] not NULL,
				lat char[12] not NULL,
				lng char[12] not NULL,
				feature text not NULL,
				filename text not NULL
			);''')
		self.conn.commit()
		print("Created")

	def select(self):
		c = self.conn.cursor()
		cursor = c.execute('''select id, time, lat, lng, feature, filename from RemoteSensing;''')
		results = []
		for row in cursor:
			results.append(row)
		return results
	
	def select_id(self, idx):
		c = self.conn.cursor()
		cursor = c.execute('''select id, time, lat, lng, feature, filename from RemoteSensing where ID=%d;'''%idx)
		results = []
		for row in cursor:
			results.append(row[-1])
		return results[0]
	
	def delete(self, idx):
		c = self.conn.cursor()
		cursor = c.execute('''delete from RemoteSensing where ID=%d''' %idx)
		self.conn.commit()

def testDB():
	db = DBConnector()
	#db.createTable()
	db.insert(1, '2020-01-01', '40.02.23N', '116.20.23E', [128 for i in range(128)], 'testimage.jpg')
	db.select()
	print(db.select_id(1))
	db.delete(1)
	db.select()

if __name__ == "__main__":
	testDB()
