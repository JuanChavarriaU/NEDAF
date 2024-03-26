import stat
from PyQt6.QtWidgets import QWidget, QTreeView, QVBoxLayout
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import QObject, Qt, QAbstractItemModel, QModelIndex
import paramiko
class FileExplorerWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create a layout
        layout = QVBoxLayout()
        
        # Create a file system model
        self.model = QFileSystemModel()
        self.model.setRootPath("/")

        # Create a tree view and set the model
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index("/"))

        # Connect double-click event to a custom slot
        self.tree_view.doubleClicked.connect(self.itemDoubleClicked)

        # Add the tree view to the layout
        layout.addWidget(self.tree_view)

        # Set the layout for the widget
        self.setLayout(layout)


    def itemDoubleClicked(self, index):
        # Get the file path of the item that was double-clicked
        file_path = self.model.filePath(index)
        print("Double-clicked file:", file_path)
        # Here you can implement actions like opening files or navigating into directories

class SSHFileSystemModel(QAbstractItemModel): #necesitamos conectar a un servidor ssh, problemas con el stfp
    def __init__(self, host, user, password):
        super().__init__()

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=host, username=user, password=password) 

        self.stfp = self.ssh.open_sftp()
        self.root = ["/", self.get_children("/")]

    def get_children(self, path):
        with self.sftp.chdir(path):
            return [(entry.filename, self.get_children(entry.filename)) for entry in self.sftp.listdir_attr() if stat.S_ISDIR(entry.st_mode)]
        
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return len(parent.internalPointer()[1])
        else:
            return len(self.root[1])

    def columnCount(self, parent=QModelIndex()):
        return 1

    def index(self, row, column, parent=QModelIndex()):
        if parent.isValid():
            data = parent.internalPointer()[1][row]
        else:
            data = self.root[1][row]
        return self.createIndex(row, column, data)

    def parent(self, index):
        return QModelIndex()  # This needs to be implemented

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return index.internalPointer()[0]       