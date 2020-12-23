"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
"""
Unit tests.
"""
import copy
import difflib
import json
import os
import random
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from moz_sp import add_join_condition
from moz_sp import parse
from moz_sp import check_schema_consistency
from moz_sp import denormalize
from moz_sp import eo_parse
from moz_sp import extract_foreign_keys, extract_values
from moz_sp import format
from moz_sp import restore_clause_order
from moz_sp import tokenize
from moz_sp.debugs import DEBUG
from src.data_processor.schema_loader import load_schema_graphs_spider, load_schema_graphs_wikisql
from src.data_processor.vocab_utils import functional_token_index
import src.utils.trans.bert_utils as bu


complex_queries = [
    "SELECT rid FROM routes WHERE dst_apid IN (SELECT apid FROM airports WHERE country  =  'United States') AND src_apid IN (SELECT apid FROM airports WHERE country  =  'United States')",
    'SELECT t3.name FROM publication_keyword AS t4 JOIN keyword AS t1 ON t4.kid  =  t1.kid JOIN publication AS t2 ON t2.pid  =  t4.pid JOIN journal AS t3 ON t2.jid  =  t3.jid WHERE t1.keyword  =  "Relational Database" GROUP BY t3.name HAVING COUNT ( DISTINCT t2.title )  >  60',
    'SELECT count(DISTINCT state) FROM college WHERE enr  <  (SELECT avg(enr) FROM college)',
    'SELECT DISTINCT T1.LName FROM STUDENT AS T1 JOIN VOTING_RECORD AS T2 ON T1.StuID  =  PRESIDENT_Vote EXCEPT SELECT DISTINCT LName FROM STUDENT WHERE Advisor  =  "2192"',

    'SELECT Roles.role_description , count(Employees.employee_id) FROM ROLES JOIN Employees ON Employees.role_code = Roles.role_code GROUP BY Employees.role_code HAVING count(Employees.employee_id)  >  1',
    'SELECT Ref_Document_Status.document_status_description FROM Ref_Document_Status JOIN Documents ON Documents.document_status_code = Ref_Document_Status.document_status_code WHERE Documents.document_id = 1',
    'SELECT Ref_Shipping_Agents.shipping_agent_name FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code WHERE Documents.document_id = 2',
    'SELECT count(*) FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code WHERE Ref_Shipping_Agents.shipping_agent_name = "USPS"',
    'SELECT Ref_Shipping_Agents.shipping_agent_name , count(Documents.document_id) FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code GROUP BY Ref_Shipping_Agents.shipping_agent_code ORDER BY count(Documents.document_id) DESC LIMIT 1',
    'SELECT Addresses.address_details FROM Addresses JOIN Documents_Mailed ON Documents_Mailed.mailed_to_address_id = Addresses.address_id WHERE document_id = 4',
    'SELECT campusfee FROM campuses AS T1 JOIN csu_fees AS T2 ON T1.id  =  t2.campus WHERE t1.campus  =  "San Jose State University" AND T2.year  =  1996',
    'SELECT campusfee FROM campuses AS T1 JOIN csu_fees AS T2 ON T1.id  =  t2.campus WHERE t1.campus  =  "San Francisco State University" AND T2.year  =  1996',
    'SELECT T1.campus FROM campuses AS t1 JOIN enrollments AS t2 ON t1.id  =  t2.campus WHERE t2.year  =  1956 AND totalenrollment_ay  >  400 AND FTE_AY  >  200',
    'SELECT degrees FROM campuses AS T1 JOIN degrees AS T2 ON t1.id  =  t2.campus WHERE t1.campus  =  "San Jose State University" AND t2.year  =  2000',
    'SELECT degrees FROM campuses AS T1 JOIN degrees AS T2 ON t1.id  =  t2.campus WHERE t1.campus  =  "San Francisco State University" AND t2.year  =  2001',
    'SELECT T1.campus FROM campuses AS t1 JOIN faculty AS t2 ON t1.id  =  t2.campus WHERE t2.faculty  >=  600 AND t2.faculty  <=  1000 AND T1.year  =  2004',
    'SELECT T2.faculty FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  t2.campus JOIN degrees AS T3 ON T1.id  =  t3.campus AND t2.year  =  t3.year WHERE t2.year  =  2002 ORDER BY t3.degrees DESC LIMIT 1',
    'SELECT T2.faculty FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  t2.campus JOIN degrees AS T3 ON T1.id  =  t3.campus AND t2.year  =  t3.year WHERE t2.year  =  2001 ORDER BY t3.degrees LIMIT 1',

    'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1',
    "SELECT name_first ,  name_last FROM player WHERE death_year = ''",
    "SELECT count(*) FROM player WHERE birth_country = 'USA' AND bats  =  'R'",
    "SELECT count(*) FROM ( SELECT * FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_winner  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' UNION SELECT * FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_loser  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' ) AS T0",
    # 'SELECT document_name FROM documents GROUP BY document_type_code ORDER BY count(*) DESC LIMIT 3 INTERSECT SELECT document_name FROM documents GROUP BY document_structure_code ORDER BY count(*) DESC LIMIT 3',

    'SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN (SELECT T2.dormid FROM dorm AS T3 JOIN has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC LIMIT 1)',
    "SELECT name ,  trade_name FROM medicine EXCEPT SELECT T1.name ,  T1.trade_name FROM medicine AS T1 JOIN medicine_enzyme_interaction AS T2 ON T2.medicine_id  =  T1.id JOIN enzyme AS T3 ON T3.id  =  T2.enzyme_id WHERE T3.product  =  'Protoporphyrinogen IX'",

    'SELECT Ref_Document_Status.document_status_description FROM Ref_Document_Status JOIN Documents ON Documents.document_status_code = Ref_Document_Status.document_status_code WHERE Documents.document_id = 1',
    'SELECT Addresses.address_details FROM Addresses JOIN Documents_Mailed ON Documents_Mailed.mailed_to_address_id = Addresses.address_id WHERE document_id = 4',
    'SELECT document_id FROM Documents WHERE document_status_code  =  "done" AND document_type_code = "Paper" INTERSECT SELECT document_id FROM Documents JOIN Ref_Shipping_Agents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code WHERE Ref_Shipping_Agents.shipping_agent_name = "USPS"',
    'SELECT employee_name FROM Employees EXCEPT SELECT Employees.employee_name FROM Employees JOIN Circulation_History ON Circulation_History.employee_id = Employees.employee_id',
    'SELECT Employees.employee_name , count(*) FROM Employees JOIN Circulation_History ON Circulation_History.employee_id = Employees.employee_id GROUP BY Circulation_History.document_id , Circulation_History.draft_number , Circulation_History.copy_number ORDER BY count(*) DESC LIMIT 1',
    "SELECT count(*) FROM Restaurant JOIN Type_Of_Restaurant ON Restaurant.ResID =  Type_Of_Restaurant.ResID JOIN Restaurant_Type ON Type_Of_Restaurant.ResTypeID = Restaurant_Type.ResTypeID GROUP BY Type_Of_Restaurant.ResTypeID HAVING Restaurant_Type.ResTypeName = 'Sandwich'",
    'SELECT sum(Spent) FROM Student JOIN Visits_Restaurant ON Student.StuID = Visits_Restaurant.StuID WHERE Student.Fname = "Linda" AND Student.Lname = "Smith"',
    'SELECT count(*) FROM Student JOIN Visits_Restaurant ON Student.StuID = Visits_Restaurant.StuID JOIN Restaurant ON Visits_Restaurant.ResID = Restaurant.ResID WHERE Student.Fname = "Linda" AND Student.Lname = "Smith" AND Restaurant.ResName = "Subway"',

    "SELECT Hosts FROM farm_competition WHERE Theme !=  'Aliens'",
    'SELECT count(*) FROM Student JOIN Visits_Restaurant ON Student.StuID = Visits_Restaurant.StuID JOIN Restaurant ON Visits_Restaurant.ResID = Restaurant.ResID WHERE Student.Fname = "Linda" AND Student.Lname = "Smith" AND Restaurant.ResName = "Subway"',

    'SELECT avg(age) FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid)',
    'SELECT count(*) FROM department WHERE department_id NOT IN (SELECT department_id FROM management)',
    'SELECT product FROM product WHERE product != (SELECT max_page_size FROM product GROUP BY max_page_size ORDER BY count(*) DESC LIMIT 1)',
    "SELECT count(*) FROM ( SELECT * FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_winner  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' UNION SELECT * FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_loser  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' ) AS T0",

    'SELECT TYPE FROM vocals AS T1 JOIN band AS T2 ON T1.bandmate  =  T2.id WHERE firstname  =  "Solveig" GROUP BY TYPE ORDER BY count(*) DESC LIMIT 1',
    'SELECT Roles.role_description , count(Employees.employee_id) FROM ROLES JOIN Employees ON Employees.role_code = Roles.role_code GROUP BY Employees.role_code HAVING count(Employees.employee_id)  >  1',
    'SELECT Ref_Document_Status.document_status_description FROM Ref_Document_Status JOIN Documents ON Documents.document_status_code = Ref_Document_Status.document_status_code WHERE Documents.document_id = 1',
    'SELECT Ref_Shipping_Agents.shipping_agent_name FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code WHERE Documents.document_id = 2',
    'SELECT count(*) FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code WHERE Ref_Shipping_Agents.shipping_agent_name = "USPS"',
    'SELECT Ref_Shipping_Agents.shipping_agent_name , count(Documents.document_id) FROM Ref_Shipping_Agents JOIN Documents ON Documents.shipping_agent_code = Ref_Shipping_Agents.shipping_agent_code GROUP BY Ref_Shipping_Agents.shipping_agent_code ORDER BY count(Documents.document_id) DESC LIMIT 1',
    'SELECT Addresses.address_details FROM Addresses JOIN Documents_Mailed ON Documents_Mailed.mailed_to_address_id = Addresses.address_id WHERE document_id = 4',
    'SELECT campusfee FROM campuses AS T1 JOIN csu_fees AS T2 ON T1.id  =  t2.campus WHERE t1.campus  =  "San Jose State University" AND T2.year  =  1996',
    'SELECT campusfee FROM campuses AS T1 JOIN csu_fees AS T2 ON T1.id  =  t2.campus WHERE t1.campus  =  "San Francisco State University" AND T2.year  =  1996',
    'SELECT T1.campus FROM campuses AS t1 JOIN enrollments AS t2 ON t1.id  =  t2.campus WHERE t2.year  =  1956 AND totalenrollment_ay  >  400 AND FTE_AY  >  200',
    'SELECT degrees FROM campuses AS T1 JOIN degrees AS T2 ON t1.id  =  t2.campus WHERE t1.campus  =  "San Jose State University" AND t2.year  =  2000',
    'SELECT degrees FROM campuses AS T1 JOIN degrees AS T2 ON t1.id  =  t2.campus WHERE t1.campus  =  "San Francisco State University" AND t2.year  =  2001',
    'SELECT T1.campus FROM campuses AS t1 JOIN faculty AS t2 ON t1.id  =  t2.campus WHERE t2.faculty  >=  600 AND t2.faculty  <=  1000 AND T1.year  =  2004',
    'SELECT T2.faculty FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  t2.campus JOIN degrees AS T3 ON T1.id  =  t3.campus AND t2.year  =  t3.year WHERE t2.year  =  2002 ORDER BY t3.degrees DESC LIMIT 1',
    'SELECT T2.faculty FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  t2.campus JOIN degrees AS T3 ON T1.id  =  t3.campus AND t2.year  =  t3.year WHERE t2.year  =  2001 ORDER BY t3.degrees LIMIT 1',
    'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1'

]


def read_sql_from_file(in_sql):
    with open(in_sql) as f:
        return f.read().strip()


def test_parse():
    sql = sys.argv[1]
    sql = 'from covid_19_cases where covid_19_cases.Date = UNK select count (*)'
    # sql = 'SELECT first_name FROM Professionals UNION SELECT first_name FROM Owners EXCEPT SELECT name FROM Dogs'
    # sql = 'SELECT T1.engineer_id ,  T1.first_name ,  T1.last_name FROM Maintenance_Engineers AS T1 JOIN Engineer_Visits AS T2 GROUP BY T1.engineer_id ORDER BY count(*) DESC LIMIT 1'
    # sql = 'select count (*) from covid_19_cases where covid_19_cases.Date = <UNK>'
    ast = eo_parse(sql)
    print(json.dumps(ast, indent=4))


def foreign_key_unit_test(sql_query, schema, ast=None):
    if ast is None:
        if isinstance(sql_query, dict):
            ast = sql_query
        else:
            ast = parse(sql_query)
    if DEBUG:
        print(sql_query)
    ast, _ = denormalize(ast, schema, return_parse_tree=True)
    foreign_keys_readable, foreign_keys = extract_foreign_keys(ast, schema)
    if DEBUG:
        print(json.dumps(ast, indent=4))
        print(json.dumps(foreign_keys_readable, indent=4))
        print(json.dumps(foreign_keys, indent=4))
        print(json.dumps(schema.foreign_keys, indent=4))
        if not (all(x in schema.foreign_keys for x in foreign_keys)):
            print(schema.name)
            import pdb
            pdb.set_trace()


def value_extractor_unit_test(sql_query, schema, ast=None):
    if ast is None:
        if isinstance(sql_query, dict):
            ast = sql_query
        else:
            ast = parse(sql_query)
    print(sql_query)
    # print(json.dumps(ast, indent=4))
    ast, _ = denormalize(ast, schema, return_parse_tree=True)
    values = extract_values(ast, schema)
    print(json.dumps(list(set(values)), indent=4))


def denormalizer_unit_test(sql_query, schema, ast=None, idx=None):
    if ast is None:
        if isinstance(sql_query, dict):
            ast = sql_query
        else:
            ast = parse(sql_query)
    dn_sql_query, contains_self_join = denormalize(ast, schema, return_parse_tree=True)
    dn_sql_tokens = tokenize(dn_sql_query, schema=schema, keep_singleton_fields=True, parsed=True,
                             value_tokenize=bu.tokenizer.tokenize)
    if DEBUG:
        print(sql_query)
        print(json.dumps(ast, indent=4))
        print(list(zip(*dn_sql_tokens)))
        print()
        import pdb
        pdb.set_trace()
    # test_passed = False
    # if re.match(alias_pattern, dn_sql_query.replace(DERIVED_TABLE_PREFIX + 'alias', '').replace(DERIVED_FIELD_PREFIX + 'alias', '')):
    #     # check self-join
    #     if not contains_self_join:
    #         raise ValueError('denormalization error')
    # n_sql_query = normalize(dn_sql_query, schema, skip={'add_semicolon', 'capitalise', 'order_query'})
    # if equal_ignoring_trivial_diffs(n_sql_query, sql_query):
    #     test_passed = True

    # idx_str = '' if idx is None else '{} '.format(idx)
    # if test_passed:
    #     print('test {}passed'.format(idx_str))
    # else:
    #     print('test {}failed'.format(idx_str))
    #     print('--------------------------')
    #     print('canonical:\t{}\n\n'.format(n_sql_query))
    #     print('original:\t{}\n\n'.format(sql_query))
    #     print('denormalized:\t{}'.format(dn_sql_query))
    #     print('--------------------------\n')
    # return test_passed


def test_execution_order():
    in_sql = sys.argv[1]
    in_sql = "SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)"
    data_dir = sys.argv[2]
    db_name = sys.argv[3]
    schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
    in_sqls = complex_queries[:4]
    db_names = ['flight_4', 'academic', 'baseball_1', 'voter_2']
    for db_name, in_sql in zip(db_names, in_sqls):
        schema = schema_graphs[db_name]
        ast = parse(in_sql)
        # print(json.dumps(ast, indent=4))
        ast_c = copy.deepcopy(ast)
        eo_sql = format(ast, schema, in_execution_order=True)
        eo_tokens = tokenize(in_sql, bu.tokenizer.tokenize, schema=schema, in_execution_order=True)
        print('in_sql: {}'.format(in_sql))
        print('eo_sql: {}'.format(eo_sql))
        # print('eo_tokens: {}'.format(eo_tokens))
        eo_ast = eo_parse(eo_sql)
        assert(json.dumps(ast_c, sort_keys=True) == json.dumps(eo_ast, sort_keys=True))
        # print(json.dumps(eo_ast, indent=4))
        restored_sql = format(eo_ast, schema)
        # print('restored_sql: {}'.format(restored_sql))
        print()


def test_restore_clause_order():
    in_sql = 'from (from countrylanguage where countrylanguage.Language = "spanish" select max (countrylanguage.Percentage)) as T0 JOIN countrylanguage ON T0. = countrylanguage.Percentage select count (*)'
    in_sql = 'from poker_player where poker_player.Earnings = (from poker_player select sum (poker_player.Earnings)) select poker_player.Money_Rank order by poker_player.Earnings desc limit 1'
    in_sql = 'from Student join Has_Pet on Student.StuID = Has_Pet.StuID join Pets on Has_Pet.PetID = Pets.PetID where Pets.PetType = "cat" select Student.Major , Student.Major'
    data_dir = sys.argv[1]
    db_name = sys.argv[2]
    schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
    schema = schema_graphs[db_name]
    print('eo_sql: {}'.format(in_sql))
    restored_sql = restore_clause_order(in_sql, schema)
    print('restored_sql: {}'.format(restored_sql))


def test_schema_consistency():
    data_dir = sys.argv[1]
    db_name = 'flight_2'
    schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
    schema = schema_graphs[db_name]
    schema.pretty_print()

    in_sql = 'SELECT singer.Name FROM concert JOIN singer_in_concert ON singer_in_concert.Singer_ID = singer.Singer_ID WHERE concert.Year = 2014'
    in_sql = 'SELECT singer.concert FROM singer WHERE singer.age  >  (SELECT avg(singer.age) FROM singer)'
    in_sql = 'SELECT singer.Name, singer.Country FROM singer INTERSECT SELECT singer.Name, singer.Country, singer.Age FROM singer WHERE singer.Age = "?" ORDER BY singer.Age DESC'
    in_sql = 'SELECT COUNT(*) FROM singer'
    in_sql = 'SELECT concert.concert_Name, concert.Theme, COUNT(*) FROM concert GROUP BY concert.Theme, concert.Theme'
    in_sql = 'SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id JOIN concert AS T3 ON T1.concert_id  =  T3.concert_id WHERE T3.year  =  2014'
    in_sql = 'SELECT AIRPORTS.AirportCode FROM AIRPORTS JOIN FLIGHTS ON AIRPORTS.AirportCode = FLIGHTS.DestAirport OR AIRPORTS.AirportCode = FLIGHTS.SourceAirport GROUP BY AIRPORTS.AirportCode ORDER BY COUNT(*) DESC LIMIT 1'

    # in_sql = 'from singer select singer.Name , singer.Country union from singer where singer.Age = "age" select singer.Name , singer.Country , singer.Age order by singer.Age desc'
    # in_sql = 'from stadium join concert on stadium.Stadium_ID = concert.Stadium_ID where stadium.Capacity = (from stadium select max (stadium.Capacity)) select count (*)'
    # in_sql = 'from singer join singer_in_concert on singer.Singer_ID = singer_in_concert.Singer_ID join concert on singer_in_concert.Singer_ID = concert.concert_ID where concert.Year = 2014 select singer.Name'
    # in_sql = 'from Students select Students.other_student_details order by Students.other_student_details desc limit  <UNK>'
    # in_sql = 'from Sections join Sections on Addresses.address_id = * where Sections.section_name = "h" select Sections.section_name'
    # in_sql = 'from stadium join concert on stadium.Stadium_ID = concert.Stadium_ID where concert.Year = 2014 select stadium.Name , stadium.Location intersect from concert join stadium on stadium.Stadium_ID = stadium.Stadium_ID where concert.Year = 2015 select stadium.Name , stadium.Location'

    # in_sql, _ = denormalize(in_sql, schema)
    ast = parse(in_sql)
    check_schema_consistency(ast, schema, verbose=True)
    # ast = eo_parse(in_sql)
    # print(restore_clause_order(in_sql, schema, schema_consistency=True))


def test_value_extractor():
    in_sql = 'SELECT singer.Name, singer.Country FROM singer INTERSECT SELECT singer.Name, singer.Country, singer.Age FROM singer WHERE singer.Age = "?" ORDER BY singer.Age DESC'
    in_sql = 'SELECT avg(age) FROM Student WHERE StuID IN ( SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "animal" INTERSECT SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "animal")'
    in_sql = 'SELECT DISTINCT T1.age FROM management AS T2 JOIN head AS T1 ON T1.head_id  =  T2.head_id WHERE T2.temporary_acting  =  \'Yes\''
    in_sql = 'SELECT t3.title FROM authors AS t1 JOIN authorship AS t2 ON t1.authid  =  t2.authid JOIN papers AS t3 ON t2.paperid  =  t3.paperid JOIN inst AS t4 ON t2.instid  =  t4.instid WHERE t4.country  =  \"USA\" AND t2.authorder  =  2 AND t1.lname  =  \"Turon\"'
    data_dir = sys.argv[1]
    db_name = sys.argv[2]
    schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
    schema = schema_graphs[db_name]
    value_extractor_unit_test(in_sql, schema)


def test_no_join_tokenizer():
    # for sql in complex_queries:
    if True:
        sql = 'SELECT avg(age) FROM Student WHERE StuID IN ( SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "food" INTERSECT SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "animal")'
        sql = 'SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN VISITORS AS T2 JOIN VISITS AS T3 ON T1.Tourist_Attraction_ID  =  T3.Tourist_Attraction_ID AND T2.Tourist_ID  =  T3.Tourist_ID WHERE T2.Tourist_Details  =  "Vincent" INTERSECT SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN VISITORS AS T2 JOIN VISITS AS T3 ON T1.Tourist_Attraction_ID  =  T3.Tourist_Attraction_ID AND T2.Tourist_ID  =  T3.Tourist_ID WHERE T2.Tourist_Details  =  "Marcelle"'
        print(sql)
        print(tokenize(sql, bu.tokenizer.tokenize, in_execution_order=True)[0])
        tokens = tokenize(sql, bu.tokenizer.tokenize, no_join_condition=True, in_execution_order=True)[0]
        sql_njc = bu.tokenizer.convert_tokens_to_string(tokens)
        print(tokens)
        print(sql_njc)
        ast_njc = eo_parse(sql_njc)
        print(json.dumps(ast_njc, indent=4))
        print()
        import pdb
        pdb.set_trace()


def test_tokenizer():
    # for sql in complex_queries:
    if True:
        sql = 'SELECT avg(age) FROM Student WHERE StuID IN ( SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "food" INTERSECT SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  "animal")'
        sql = 'SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN VISITORS AS T2 JOIN VISITS AS T3 ON T1.Tourist_Attraction_ID  =  T3.Tourist_Attraction_ID AND T2.Tourist_ID  =  T3.Tourist_ID WHERE T2.Tourist_Details  =  "Vincent" INTERSECT SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN VISITORS AS T2 JOIN VISITS AS T3 ON T1.Tourist_Attraction_ID  =  T3.Tourist_Attraction_ID AND T2.Tourist_ID  =  T3.Tourist_ID WHERE T2.Tourist_Details  =  "Marcelle"'
        sql = "SELECT Perpetrator_ID FROM perpetrator WHERE Year IN ('1995.0', '1994.0', '1982.0')"
        print(sql)
        data_dir = sys.argv[1]
        db_name = sys.argv[2]
        schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
        schema = schema_graphs[db_name]
        tokens = tokenize(sql, bu.tokenizer.tokenize, in_execution_order=True, schema=schema)[0]
        print(tokens)
        print()

def test_atomic_tokenizer():
    for sql in complex_queries:
        tokens, token_types, constants = tokenize(sql, bu.tokenizer.tokenize, atomic_value=True,
                                                  num_token=functional_token_index['num_token'],
                                                  str_token=functional_token_index['str_token'])
        print(sql)
        print(tokens)
        print(token_types)
        for constant in constants:
            print(constant)
        print()
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    # test_denormalizer()
    # test_normalizer()
    # test_constant_extractor()
    # test_execution_order()
    # test_restore_clause_order()
    test_parse()
    # test_schema_consistency()
    # test_no_join_tokenizer()
    # test_tokenizer()
    # test_tokenizer()
    # test_atomic_tokenizer()
    # test_value_extractor()
