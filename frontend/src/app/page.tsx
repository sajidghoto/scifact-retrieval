
export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center gap-4 px-6">
      <p className="text-sm font-medium uppercase tracking-widest text-gray-500">
        Research Engineering
      </p>

      <h1 className="text-4xl font-bold tracking-tight">
        SciFact Retrieval System
      </h1>

      <p className="max-w-xl text-gray-600">
        A reproducible document-ingestion and retrieval benchmark.
        Dataset ingestion, baseline chunking, and evaluation will
        be introduced incrementally.
      </p>

      <p className="text-sm text-gray-500">
        Initial skeleton — retrieval is not implemented yet.
      </p>
    </main>
  );
}
