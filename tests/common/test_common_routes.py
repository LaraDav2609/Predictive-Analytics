import unittest

from common.api.common_routes import list_sports
from common.models.sport import Sport


class CommonRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseball_card_links_to_baseball_section(self):
        sports = await list_sports()
        baseball = next(sport for sport in sports if sport.sport == Sport.BASEBALL)

        self.assertEqual("/Sports#baseball", baseball.href)


if __name__ == "__main__":
    unittest.main()
