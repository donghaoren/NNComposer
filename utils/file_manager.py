import os
import shutil
import uuid

class FileManager:
    def __init__(self):
        pass
        
    def getTemporaryFile(self):
        """ Get a path to a temporary file """
        raise NotImplemented
    
    def upload(self, source, destination):
        """ Upload file stored in 'source' to destination """
        raise NotImplemented
    
    def download(self, filename):
        """ Download a file, return the temporary file name """
        raise NotImplemented
        
        
    def getCheckpointFile(self, modelName, sessionTimestamp, epoch, loss):
        return "%s/%d-%06d-%.8f.h5" % (modelName, sessionTimestamp, epoch, loss)
    
    def getInterruptFile(self, modelName, sessionTimestamp, timestamp):
        return "%s/%d-interrupt-%d.h5" % (modelName, sessionTimestamp, timestamp)
    
    def saveModel(self, model, path):
        tmp_file = self.getTemporaryFile()
        model.save(tmp_file)
        self.upload(tmp_file, path)
        os.unlink(tmp_file)
        
    def loadModel(self, model, path):
        tmp_file = self.download(path)
        model.load_weights(tmp_file)
        os.unlink(tmp_file)
    
class FileManagerFS(FileManager):
    def __init__(self, output_dir = "output"):
        FileManager.__init__(self)
        self.output_dir = output_dir
        
    def ensureDirectory(self, directory):
        path = os.path.join(self.output_dir, directory)
        try: os.makedirs(path)
        except: pass
    
    def getTemporaryFile(self):
        self.ensureDirectory("temporary")
        return os.path.join(self.output_dir, "temporary", str(uuid.uuid4()))
    
    def upload(self, source, destination):
        self.ensureDirectory(os.path.dirname(destination))
        shutil.copyfile(source, os.path.join(self.output_dir, destination))
        
    def download(self, filename):
        temp_file = self.getTemporaryFile()
        shutil.copyfile(os.path.join(self.output_dir, filename), temp_file)
        return temp_file