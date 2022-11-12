import requests
from typing import Tuple


class RequestHandler:
    @staticmethod
    def process_response(req: requests.Response) -> Tuple[bool, bytes]:
        """
        This method is used to handle the http responses.

        :param req: request object
        :return: bool of the success state and the data
        """
        if req.status_code == 406 or req.status_code == 401:
            return False, req.content
        elif req.status_code == 200:
            return True, req.content
        else:
            return False, req.content


request_handler = RequestHandler()
