#!/usr/bin/env python
# coding: utf-8


import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from Bio import Entrez


def get_htmldoc(url):
    r = requests.get(url)
    if r.status_code==200:
        return r.text
    else:
        raise ValueError(f'status code:{r.status_code}')


def get_text_header(tag):                
    return tag.get_text().strip()

def get_text_rows(tag):
    accession_class = ['genbank','ref','uniprot']
    if tag['class'][0] in accession_class:
        if tag.text:
            # find if there are multiple accession numbers
            get_a_tags = tag.find_all('a')
            return ','.join([a_tag['href'] for a_tag in get_a_tags])
                
    return tag.get_text().strip()


def get_headers(tag):
    header_coltags = tag.find_all('th')
    return tuple(map(get_text_header,header_coltags))

def get_rows(tag):
    row_coltags = tag.find_all('td')
    return tuple(map(get_text_rows,row_coltags))


def parse_html(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    main_table = soup.find_all('table')[1]
    table_rows = main_table.find_all('tr')[1:]
    header_row = table_rows[0]
    content_rows = table_rows[1:]
    header = get_headers(header_row)
    all_rows = list(map(get_rows,content_rows))
    assert list(map(len,all_rows)) == [len(header) for i in range(len(all_rows))]
    df = pd.DataFrame(all_rows,columns=header).replace('','missing')
    return df   


def parse_fasta(fasta):
    header = []
    sequences = []
    curr_sequence = ''
    for lines in fasta:
        if lines.startswith('>'):
            header.append(lines)
            if curr_sequence:
                sequences.append(curr_sequence)
                curr_sequence = ''
        else:
            curr_sequence += lines

    sequences.append(curr_sequence)
    return header,sequences


def get_seq_genbank(genbank_url):
    pattern = re.compile('^http://www.ncbi.nlm.nih.gov/protein/(.+)')
    m = re.match(pattern,genbank_url)
    genbank_id = m.group(1)
    Entrez.email = 'dzb5732@psu.edu'
    handle = Entrez.efetch(db='protein',id=genbank_id,rettype='fasta')
    fasta_file = []
    for lines in handle:
        fasta_file.append(lines.strip())
    handle.close()
    fasta_header,fasta_seq = parse_fasta(fasta_file)
    if len(fasta_header) == len(fasta_seq) == 1:
        return fasta_seq[0]
    else:
        return False


def get_seq_uniprot(uniprot_url):
    url = uniprot_url+'.fasta'
    r = requests.get(url)
    fasta_file = r.text.strip().split('\n')
    fasta_header,fasta_seq = parse_fasta(fasta_file)
    #check if a single fasta file has been returned
    if len(fasta_header) == len(fasta_seq) == 1:
        return fasta_seq[0]
    else:
        return False

    
def get_seq_refseq(refseq_url):
    pattern = re.compile('^http://www.ncbi.nlm.nih.gov/protein/(.+)')
    m = re.match(pattern,refseq_url)
    refseq_id = m.group(1)
    Entrez.email = 'dzb5732@psu.edu'
    handle = Entrez.efetch(db='protein',id=refseq_id,rettype='fasta')
    fasta_file = []
    for lines in handle:
        fasta_file.append(lines.strip())
    handle.close()
    fasta_header,fasta_seq = parse_fasta(fasta_file)
    if len(fasta_header) == len(fasta_seq) == 1:
        return fasta_seq[0]
    else:
        return False


def get_prot_seq(row):
    if row['UniProt'] != 'missing':
        urls = row['UniProt'].split(',')
        for url in urls:
            seq = get_seq_uniprot(url)
            if seq:
                return seq
            
    if row['GenBank ID'] != 'missing':
        urls = row['GenBank ID'].split(',')
        for url in urls:
            seq = get_seq_genbank(url)
            if seq:
                return seq

    if row['RefSeq'] != 'missing':
        urls = row['RefSeq'].split(',')
        for url in urls:
            seq = get_seq_refseq(url)
            if seq:
                return seq
    
    return False


def main():
    url_page5 = 'http://www.enzyme.cbirc.iastate.edu/?a=view&c=sequencegroup&id=31&sg_sort_column=&sg_sort_order=1&sg_page=5'
    url_page6 = 'http://www.enzyme.cbirc.iastate.edu/?a=view&c=sequencegroup&id=31&sg_sort_column=&sg_sort_order=1&sg_page=6'
    html_doc_p5 = get_htmldoc(url_page5)
    html_doc_p6 = get_htmldoc(url_page6)
    df_p5 = parse_html(html_doc_p5)
    df_p6 = parse_html(html_doc_p6)
    df_p5['AA_Sequence'] = df_p5.apply(get_prot_seq,axis=1)
    df_p6['AA_Sequence'] = df_p6.apply(get_prot_seq,axis=1)
    df = pd.concat((df_p5,df_p6))
    return df
    

if __name__=='__main__':
    df = main()
    df.to_csv('../data/thyme/thyme_dataset.csv',columns=['Sequence','Organism','AA_Sequence'],header=False,index=False)
    

