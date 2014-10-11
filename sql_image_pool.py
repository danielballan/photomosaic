import sqlite3
from image_pool import ImagePool
import logging

# Configure logger.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def connect(db_path):
    "Connect to a sqlite database at db_path. If it does not exist, create it."
    try:
        db = sqlite3.connect(db_path)
    except IOError:
        logger.error("Cannot connect to SQLite database at %s",  db_path)
        return
    db.row_factory = sqlite3.Row # Rows are dictionaries.
    return db
    
def create_tables(db):
    """Create Images for image meta info, Color for RGB values and LabColor
    for LAB values. RGB and LAB are used for different steps, RGB for levels
    adjustments are LAB for measure perceived color difference precisely.
    Thus Color and LabColor are organized somewhat differently."""
    c = db.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS Images
                 (image_id INTEGER PRIMARY KEY,
                  usages INTEGER,
                  w INTEGER,
                  h INTEGER,
                  filename TEXT UNIQUE)""")
    c.execute("""CREATE TABLE IF NOT EXISTS Colors
                 (color_id INTEGER PRIMARY KEY,
                  image_id INTEGER,
                  region INTEGER,
                  red INTEGER,
                  green INTEGER,
                  blue INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS LabColors
                 (labcolor_id INTEGER PRIMARY KEY,
                  image_id INTEGER,
                  region INTEGER,
                  L1 REAL,
                  a1 REAL,
                  b1 REAL,
                  L2 REAL,
                  a2 REAL,
                  b2 REAL,
                  L3 REAL,
                  a3 REAL,
                  b3 REAL,
                  L4 REAL,
                  a4 REAL,
                  b4 REAL)""")
    c.close()
    db.commit()


class SqlImagePool(ImagePool):
    def __init__(self, db_name):
        self.db = connect(db_name)
        create_tables(self.db)
            
    def insert(self, filename, w, h, rgb, lab):
        """Insert image info in the Images table and color information in the
        Color and LabColor tables."""
        c = self.db.cursor()
        try:
            c.execute("""INSERT INTO Images (usages, w, h, filename)
                         VALUES (?, ?, ?, ?)""",
                      (0, w, h, filename))
            image_id = c.lastrowid
            c.executemany("""INSERT INTO Colors (image_id, region, red, green, blue)
                             VALUES (?, ?, ?, ?, ?)""",
                             [tuple([image_id, region] + list(colors)) \
                              for region, colors in enumerate(rgb)])
            c.execute("""INSERT INTO LabColors (image_id,
                         L1, a1, b1, L2, a2, b2, L3, a3, b3, L4, a4, b4)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         tuple([image_id] + [ch for reg in lab for ch in reg]))
        except sqlite3.IntegrityError:
            logger.warning("Image %s is already in the table. Skipping it.",
                           filename)
        except:
            logger.warning("Unknown problem with image %s. Skipping it.",
                           filename)
        finally:
            c.close()
            
    def pool_histogram(self):
        """Generate a histogram of the images in the pool.
        Return a dictionary of the channels red, green blue.
        Each dict entry contains a list of the frequencies correspond to the
        domain 0 - 255.""" 
        hist = {}
        c = self.db.cursor()
        c.execute("SELECT * FROM sqlite_master WHERE type='table';")
        for x in c.fetchall():
            print '\n==========\n'.join(map(str, x))
            print
            print
        exit(0)
        try: 
            for ch in ['red', 'green', 'blue']:
                c.execute("""SELECT {ch}, count(*)
                             FROM Colors 
                             GROUP BY {ch}""".format(ch=ch))
                values, counts = zip(*c.fetchall())
                # Normalize the histogram to 256 for readability,
                # and fill in 0 for missing entries.
                full_domain = range(0,256)
                N = sum(counts)
                all_counts = [256./N*counts[values.index(i)] if i in values else 0 \
                              for i in full_domain]
                hist[ch] = all_counts
        finally:
            c.close()
        return hist
            
    def __contains__(self, filename):
        c = self.db.cursor()
        try: 
            c.execute("SELECT count(*) FROM Images WHERE filename=?", (filename,))
            return c.fetchone()[0] > 0
        finally:
            c.close()
        return False
        
    def __len__(self):
        c = self.db.cursor()
        try: 
            c.execute("SELECT count(*) FROM Images")
            return c.fetchone()[0] 
        finally:
            c.close()
        return 0   

    def close(self):
        self.db.commit()
        self.db.close()
