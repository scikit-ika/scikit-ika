/*
 * Copyright 2015 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package seqprediction;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import ipredict.database.Item;
import ipredict.database.Sequence;

public class SeqPredictionClient {
    private static final Logger logger = Logger.getLogger(SeqPredictionClient.class.getName());

    private final ManagedChannel channel;
    private final PredictorGrpc.PredictorBlockingStub blockingStub;

    /** Construct client connecting to HelloWorld server at {@code host:port}. */
    public SeqPredictionClient(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext()
                .build());
    }

    /** Construct client for accessing HelloWorld server using the existing channel. */
    SeqPredictionClient(ManagedChannel channel) {
        this.channel = channel;
        blockingStub = PredictorGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void predict(int seqId, int[] seq) {
        logger.info("Will try to predict the next position...");

        SequenceMessage.Builder builder = SequenceMessage.newBuilder();
        for (int i = 0; i < seq.length; i++) {
            builder.addSeq(seq[i]);
        }

        SequenceMessage request = builder.setSeqId(seqId).build();
        SequenceMessage response;

        try {
            response = blockingStub.predict(request);

        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;

        }

        logger.info("Prediction: " + response.getSeq(0));
    }

    public static void main(String[] args) throws Exception {

        // Access a service running on the local machine on port 50051
        SeqPredictionClient client = new SeqPredictionClient("localhost", 50051);

        try {
            int seqId = 1;
            int[] seq = {1, 2, 3, 4};
            client.predict(seqId, seq);

        } finally {
            client.shutdown();

        }
    }
    
}
