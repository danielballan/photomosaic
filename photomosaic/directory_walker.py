import os

class DirectoryWalker:
    def __init__(self, directory):
        self.files = []
        self.index = 0
        
        for root,dirs,files in os.walk(directory):
            for f in files:
                self.files.append( root + os.sep + f )
                
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.files[index]
