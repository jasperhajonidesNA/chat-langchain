# 🦜️🔗 Chat LangChain

This repo is an implementation of a chatbot specifically focused on question answering over the [LangChain documentation](https://python.langchain.com/).
Built with [LangChain](https://github.com/langchain-ai/langchain/), [LangGraph](https://github.com/langchain-ai/langgraph/), and [Next.js](https://nextjs.org).

Deployed version: [chat.langchain.com](https://chat.langchain.com)

> Looking for the JS version? Click [here](https://github.com/langchain-ai/chat-langchainjs).

The app leverages LangChain and LangGraph's streaming support and async API to update the page in real time for multiple users.

## Running locally

This project is now deployed using [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/), which means you won't be able to run it locally (or without a LangGraph Cloud account). If you want to run it WITHOUT LangGraph Cloud, please use the code and documentation from this [branch](https://github.com/langchain-ai/chat-langchain/tree/langserve).

> [!NOTE]
> This [branch](https://github.com/langchain-ai/chat-langchain/tree/langserve) **does not** have the same set of features.

## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site as well as the Github Codebase
2. Load html with LangChain's [RecursiveURLLoader](https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader) and [SitemapLoader](https://python.langchain.com/docs/integrations/document_loaders/sitemap)
3. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [Weaviate vectorstore wrapper](https://python.langchain.com/docs/integrations/vectorstores/weaviate) (with OpenAI's embeddings).

Question-Answering has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using an LLM.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## Documentation

Looking to use or modify this Use Case Accelerant for your own needs? We've added a few docs to aid with this:

- **[Concepts](./CONCEPTS.md)**: A conceptual overview of the different components of Chat LangChain. Goes over features like ingestion, vector stores, query analysis, etc.
- **[Modify](./MODIFY.md)**: A guide on how to modify Chat LangChain for your own needs. Covers the frontend, backend and everything in between.
- **[LangSmith](./LANGSMITH.md)**: A guide on adding robustness to your application using LangSmith. Covers observability, evaluations, and feedback.
- **[Production](./PRODUCTION.md)**: Documentation on preparing your application for production usage. Explains different security considerations, and more.
- **[Deployment](./DEPLOYMENT.md)**: How to deploy your application to production. Covers setting up production databases, deploying the frontend, and more.

## Deploying the frontend to GCP

While the recommended hosting provider is Vercel, you can also run the Next.js
frontend on Google Cloud Run. A `Dockerfile` is available in
`./frontend` for this purpose. To deploy:

```bash
cd frontend
gcloud builds submit --tag gcr.io/<PROJECT_ID>/chat-langchain-frontend
gcloud run deploy chat-langchain-frontend \
  --image gcr.io/<PROJECT_ID>/chat-langchain-frontend \
  --region <REGION> --platform managed --allow-unauthenticated \
  --set-env-vars NEXT_PUBLIC_API_URL=<backend-url>,API_BASE_URL=<backend-url>,LANGCHAIN_API_KEY=<langsmith-key>
```

Replace the placeholders with your own project information and environment
variables. After deployment completes, Cloud Run will provide a public URL for
your app.

### Testing the Docker image locally

You can build and run the frontend Docker image before deploying:

```bash
cd frontend
docker build -t chat-langchain-frontend .
docker run -p 8080:8080 chat-langchain-frontend
```

Then open `http://localhost:8080` in your browser to test the app.

### GCP authentication and permissions

If deployment commands fail with messages like
"user is forbidden from accessing the bucket" check these points:

1. **Service Usage Consumer role** – grant your user account
   `roles/serviceusage.serviceUsageConsumer` in IAM.
2. **Organization policies** – verify there are no organization policies
   restricting Cloud Build or Storage access.
3. **Cloud Build API** – ensure it is enabled in APIs & Services.
4. **Cloud Build service account** – `[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`
   needs Storage permissions (e.g. Storage Object Admin).
5. **Billing and quotas** – confirm billing is active and quotas are OK.
6. **Refresh credentials** – run `gcloud auth login` and
   `gcloud auth application-default login`.

Use `gcloud auth list` to verify which accounts are active before retrying the
deployment commands.
