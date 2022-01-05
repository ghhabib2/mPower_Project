from data_loader.data_loader import DataLoader


class TappingDataLoader(DataLoader):
    """ Load the Tapping data based on the Query """

    def load_data(self, limit):
        # Query mPoser Project
        table = self.syn.tableQuery(f"""
        SELECT  *
        FROM syn5511439
        LIMIT {limit}
        """)

        # Convert to the DataFrame
        df = table.asDataFrame()

        return df
