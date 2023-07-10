from itemadapter import ItemAdapter
import json
import wget
import uuid
from tmg.conf import PROJECT_ROOT_PATH, DATA_CRAWLER_ROOT_PATH

class MusicCrawlPipeline:
    def open_spider(self) -> None:
        self.file = open("items.jsonl", "w")

    def close_spider(self) -> None:
        self.file.close()

    def process_item(self, item):
        """
        process_item Dump the crawling result to a json file and download all the mp3 files.

        Parameters
        ----------
        item : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        music_file_path = str(uuid.uuid4()) + ".mp3"
        music_file_path_parrent = DATA_CRAWLER_ROOT_PATH / 'mp3--'
        music_file_path_parrent.mkdir(exist_ok=True)
        music_file_path = music_file_path_parrent / music_file_path
        wget.download(item["link"], music_file_path)
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file.write(line)
        return item
