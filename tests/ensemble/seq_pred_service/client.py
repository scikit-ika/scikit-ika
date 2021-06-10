#!/usr/bin/env python

from __future__ import print_function
import logging

import grpc

import seqprediction_pb2
import seqprediction_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:

        stub = seqprediction_pb2_grpc.PredictorStub(channel)
        seqs = [[17, 16, 15, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [17, 16, 15, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [17, 16, 15, 14, 13, 12, 11, 10]]

        for idx, s in enumerate(seqs):
            print(s[:-1])

            response = stub.predict(seqprediction_pb2.SequenceMessage(seqId=idx, seq=s[:-1]))
            result = ",".join([str(v) for v in response.seq])
            # print(f"seqID = {response.seqId}")
            print(f"Predicted sequence: {result}" )

            if stub.train(seqprediction_pb2.SequenceMessage(seqId=idx, seq=s)).result:
                print("Training completed\n")
            else:
                print("Training failed\n")

if __name__ == '__main__':
    logging.basicConfig()
    run()
