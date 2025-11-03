from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path
from enum import Enum
import json

# Identifier enums for variants for index
# Tailor to add specifics that are implemented
class IndexInfo(Enum):
    BOOLEAN = 1
    WORDCOUNT = 2
    TFIDF = 3
class DataStore(Enum):
    CUSTOM = 1
    DB1 = 2
    DB2 = 3
    POSTGRES = 2  # PostgreSQL GIN
    REDIS = 2     # Redis key-value store
class Compression(Enum):
    NONE = 1
    CODE = 2
    CLIB = 3
class QueryProc(Enum):
    TERMatat = 'T'
    DOCatat = 'D'
    DOCatatt = 'D'  # Alias for consistency
class Optimizations(Enum):
    Null = '0'
    SKIP = 'sp'  # Added for skip pointers
    Skipping = 'sp'
    Thresholding = 'th'
    EarlyStopping = 'es'
  
class IndexBase(ABC):
    """
    Base index class with abstract methods to inherit for specific implementations.
    """
    def __init__(self, core, info, dstore, qproc, compr, optim):
      """
      Sample usage:
          idx = IndexBase(core='ESIndex', info='BOOLEAN', dstore='DB1', compr='NONE', qproc='TERMatat', optim='Null')
          print (idx)
      """
      assert core in ('ESIndex', 'SelfIndex')
      long = [ IndexInfo[info], DataStore[dstore], Compression[compr], QueryProc[qproc], Optimizations[optim] ]
      short = [k.value for k in long]
      self.identifier_long = "core={}|index={}|datastore={}|compressor={}|qproc={}|optim={}".format(*[core]+long)
      self.identifier_short = "{}_i{}d{}c{}q{}o{}".format(*[core]+short)
        
    def __repr__(self):
        return f"{self.identifier_short}: {self.identifier_long}"
      
    @abstractmethod
    def create_index(self, index_id: str, files: Iterable[tuple[str, str]]) -> None: 
        """Creates and index for the given files"""
        pass
            
    @abstractmethod
    def load_index(self, serialized_index_dump: str) -> None:
        """Loads an already created index into memory from disk"""
        pass
        
    @abstractmethod
    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        """Updates an index"""
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        """Queries the already loaded index"""
        pass
  
    @abstractmethod
    def delete_index(self, index_id: str) -> None:
        """Deletes the index with the given index_id"""
        pass
  
    @abstractmethod
    def list_indices(self) -> Iterable[str]:
        """Lists all indices"""
        pass
  
    @abstractmethod
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """Lists all files indexed in the given index"""
        pass
