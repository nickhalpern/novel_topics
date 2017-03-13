import os
import re
import codecs
import glob
import pandas as pd


def encoding(fn):
    for line in open(fn):
        if line.startswith("Character set encoding:"):
            _, encoding = line.split(":")
            return encoding.strip()
    return "latin1"


def beautify(fn, outputdir):
    ''' Reads a raw Project Gutenberg etext, reformat paragraphs,
    and removes fluff.  Determines the title of the book and uses it
    as a filename to write the resulting output text. '''
    remove = ["Produced by","End of the Project Gutenberg","End of Project Gutenberg"]

    codecmap = {
        "latin1": "latin1",
        "ISO Latin-1": "latin1",
        "ISO-8859-1": "latin1",
        "UTF-8": "utf8",
        "ASCII": "ascii",
        }

    codec = codecmap.get(encoding(fn), "latin1")
    lines = [line.strip() for line in codecs.open(fn, "r", codec)]
    collect = False
    lookforsubtitle = False
    outlines = []
    startseen = endseen = False
    title=""
    year = ""
    author = ""
    for line in lines:
        if line.startswith("Title: "):
            title = line[7:]
            lookforsubtitle = True
            continue
        if line.startswith("Author: "):
            author = line[8:]
            if author == 'Human Genome Project':
                break
            continue
        if line.startswith("1") & (year == ""):
            if len(line) == 4: #==5:
                year = line
            if len(line) == 5:
                year = line[0:4]
                # year = line[0:4]
        if lookforsubtitle:
            if not line.strip():
                lookforsubtitle = False
            else:
                subtitle = line.strip()
                subtitle = subtitle.strip(".")
                title += ", " + subtitle
        if ("*** START" in line) or ("***START" in line) or (line.startswith("*END THE SMALL PRINT!")):
            collect = startseen = True
            paragraph = ""
            continue
        if ("*** END" in line) or ("***END" in line):
            endseen = True
            break
        if not collect:
            continue
        if not line:
            paragraph = paragraph.strip()
            for term in remove:
                if paragraph.startswith(term):
                    paragraph = ""
            if paragraph:
                outlines.append(paragraph)
                outlines.append("")
            paragraph = ""
        else:
            paragraph += " " + line


    # Compose a filename.  Replace some illegal file name characters with alternatives.
    lastpart = fn
    parts = fn.split(os.sep)
    if len(parts):
        lastpart = parts[-1]
    ofn = title[:150] + ", " + lastpart
    ofn = ofn.replace("&", "en")
    ofn = ofn.replace("/", "-")
    ofn = ofn.replace("\"", "'")
    ofn = ofn.replace(":", ";")
    ofn = ofn.replace(",,", ",")

    # Report on anomalous situations, but don't make it a showstopper.
    if not title:
        print ofn
        print "    Problem: No title found\n"
    if not startseen:
        print ofn
        print "    Problem: No '*** START' seen\n"
    if not endseen:
        print ofn
        print "    Problem: No '*** END' seen\n"

    f = codecs.open(os.path.join(outputdir, ofn), "w", "utf8")
    f.write("\n".join(outlines))
    f.close()
    
    return title, author, year



if __name__ == '__main__':

    if not os.path.exists("ebooks"):
        os.mkdir("ebooks")

    file_name_list = []
    title_list = []
    author_list = []
    year_list = []

    for fn in glob.glob("ebooks-unzipped/*.txt"):
        print '{}------'.format(fn)
        try:
            title, author, year = beautify(fn, "ebooks")
            print 'Author: {}\nTitle: {}\nYear: {}'.format(title, author, year)
            file_name_list.append(fn)
            title_list.append(title)
            author_list.append(author)
            year_list.append(year)
            # os.remove(fn)
        except:
            print 'Could not load book'

    A = pd.DataFrame({'file': file_name_list, 'title': title_list, 'author': author_list, 'year': year_list})
    A.to_csv('book_data.csv')
