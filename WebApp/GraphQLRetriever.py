import requests

class GraphQLRetriever():
    '''
    Class that helps us to format a GraphQL query to send it in a request format, using the requests package.
    We will need the endpoint of the GraphQL database, a username, and a token (if required)
    '''
    def __init__(self, endpoint, username, token):        
        self.endpoint = endpoint
        self.auth = (username, token)
    
    @staticmethod
    def format_query(query, **variables):
        '''
        Formats the query so that we can send the request as required
        
        The returned format is {"query": "query(<parameters>){ <body> }", "variables" : { <variables> }}
        
        Parameters:
        
        query: The query in GraphQL regular format. Double quotes should
                be escaped with triple invert slash (\\\\\\")
                
        variables: The variables that will be used in the query
        '''
        
        #Format should be {"query": "query(<parameters>){ <body> }", "variables" : { <variables> }}
        query = query.replace('\n', '') #We get rid of linebreaks since having them breaks the query
        query = "{\"query\": \"query {" + query + "}\"}"
        return query
    
    def post_query(self, query):   
        '''
        We attempt to perform the query with the parameters that were
        defined before.
        
        Parameters:
        
        query: The query in GraphQL regular format. Double quotes should
                be escaped with triple invert slash (\\\\\\")
        '''
        response = requests.post(self.endpoint, data=self.format_query(query), auth=self.auth)
        return response.json()